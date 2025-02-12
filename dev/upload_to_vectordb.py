import os
import sys
import json
import asyncio
import requests
import httpx
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProgramInfo:
    url: str
    title: str
    summary: str
    content: str
    chunk_number: int
    embedding: List[float]
    metadata: Dict[str, Any]

async def insert_chunk(chunk: ProgramInfo):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

import re

def process_and_modify_markdown(content: str) -> str:
    """
    Cleans and formats markdown content by:
    - Removing unwanted navigation and header sections.
    - Starting from line 81 and excluding lines 83 to 106 (inclusive).
    - Removing everything after encountering "* [ACADEMICS](#)".
    - Fixing links by keeping only the URL inside angle brackets <>.
    - Removing all lines starting with "![]".
    - Cleaning special characters and unwanted links.
    - Preserving essential markdown structure while maintaining readability.
    - Modifying the section starting with "## Take a Quick Look".
    
    Args:
        content (str): Raw markdown content.
    
    Returns:
        str: Processed and formatted markdown content.
    """
    # Split the content into lines and work from line 81 onward
    lines = content.split("\n")
    relevant_lines = lines[80:]  # Skip header section

    filtered_lines = []
    keep_paragraph = False
    special_chars = set(['@', '%', '&', '+', '=', '|', '<', '>', '^', '~', '-'])

    for index, line in enumerate(relevant_lines):
        stripped_line = line.strip()

        # Stop processing if we reach the unwanted section
        if stripped_line.startswith("Our enrollment representatives"):
            break

        # Skip specific index range (lines 83 to 106)
        if 83 <= (index + 81) <= 106:
            continue

        # Skip lines starting with special characters
        if stripped_line and stripped_line[0] in special_chars:
            continue

        # Skip image markdown
        if stripped_line.startswith("![]"):
            continue

         # Skip additioanl hyperlinks.
        if stripped_line.startswith("* ["):
            continue

        # Fix link formatting by keeping only the URL inside angle brackets
        line = re.sub(r"\[([^\]]+)\]\([^\(\)<>]*<([^<>]+)>\)", r"[\1](\2)", line)

        # Preserve and clean markdown headings
        if stripped_line.startswith("#"):
            line = re.sub(r'#+\s+', lambda m: m.group().strip() + ' ', line)
            filtered_lines.append(line)
            keep_paragraph = True

        # Preserve bullet points
        elif stripped_line.startswith("*") and len(stripped_line) > 1:
            line = re.sub(r'\*\s*', '* ', stripped_line)
            filtered_lines.append(line)
            keep_paragraph = True

        # Handle paragraph content
        elif keep_paragraph and stripped_line:
            line = re.sub(r'\\([\\`*_{}[\]()#+.!-])', r'\1', line)  # Remove escape chars
            filtered_lines.append(line)

        # Handle URLs
        elif "http" in stripped_line and not stripped_line.startswith("["):
            filtered_lines.append(line)

        # Handle empty lines
        elif stripped_line == "":
            filtered_lines.append("")
            keep_paragraph = False

    # Join lines and clean up multiple empty lines
    formatted_content = "\n".join(filtered_lines)
    formatted_content = re.sub(r'\n{3,}', '\n\n', formatted_content)

    # Modify the section starting with "## Take a Quick Look"
    pattern = r'## Take a Quick Look(.*?)(?=\n\n|$)'
    match = re.search(pattern, formatted_content, re.DOTALL)
    
    if match:
        # Extract the section content and remove extra spaces and newlines
        section_content = match.group(1).strip()
        
        # Separate lines by newlines
        lines = section_content.split('\n')
        
        # The first line is a description, so we leave it as is
        modified_section = lines[0] + "\n"
        
        # Iterate through the following lines, which should be in pairs (label, value)
        for i in range(1, len(lines), 2):
            label = lines[i].strip()
            value = lines[i + 1].strip() if i + 1 < len(lines) else ''
            
            # Format them as "Label : Value"
            if label and value:
                modified_section += f"{label} : {value}\n"
        
        # Replace the original section with the newly formatted one
        formatted_content = formatted_content.replace(match.group(0), f"## Take a Quick Look\n{modified_section}")

    return formatted_content.strip()


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from local LLama model using mxbai-embed-large."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "nomic-embed-text:latest",
                    "prompt": text
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Add validation and logging
            if not result.get("embedding"):
                print(f"Warning: Empty embedding returned. API response: {result}")
                return None
            
            embedding = result["embedding"]
            print(f"Successfully generated embedding with dimension: {len(embedding)}")
            
            return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

def chunk_program_content(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Split text content into three chunks:
    1. Content before 'course plan'
    2. Content between 'course plan' and '## Cost and Tuition'
    3. Content after '## Cost and Tuition'
    
    Args:
        text (str): Input text to be split
        
    Returns:
        List[str]: List containing three chunks of text
    """
    # Split the text into lines while preserving empty lines
    lines = text.splitlines(keepends=True)
    
    chunk1_lines = []
    chunk2_lines = []
    chunk3_lines = []
    
    # Track which chunk we're currently building
    current_chunk = 1
    
    # Process each line
    for line in lines:
        # Check for first split condition (course plan)
        if current_chunk == 1 and 'course plan' in line.lower():
            current_chunk = 2
            chunk2_lines.append(line)  # Include the splitting line in chunk 2
            continue
            
        # Check for second split condition (Cost and Tuition)
        if current_chunk == 2 and line.strip().startswith("## Cost and Tuition"):
            current_chunk = 3
            chunk3_lines.append(line)  # Include the splitting line in chunk 3
            continue
            
        # Add line to appropriate chunk
        if current_chunk == 1:
            chunk1_lines.append(line)
        elif current_chunk == 2:
            chunk2_lines.append(line)
        else:
            chunk3_lines.append(line)
    
    # Join the lines back together for each chunk
    chunk1 = ''.join(chunk1_lines)
    chunk2 = ''.join(chunk2_lines)
    chunk3 = ''.join(chunk3_lines)
    
    # If no splits were found, handle appropriately
    if not chunk2_lines and not chunk3_lines:
        return [chunk1, '', '']
    elif not chunk3_lines:
        return [chunk1, chunk2, '']
    
    return [chunk1, chunk2, chunk3]

    


async def extract_program_info(chunk: str, url: str, chunk_number: int, program_details: dict = None) -> Optional[ProgramInfo]:
    """
    Extract program information using local LLM and enhance metadata with program details.
    Maintains consistent program details across chunks while keeping chunk-specific data unique.
    """
    system_prompt = """You are a JSON-only response API that extracts program information from educational content.
    Your task is to extract a brief summary.
    RESPONSE FORMAT:
    You must return a valid JSON object with exactly this structure:
    {
        "summary": "Program summary here"
    }
    RULES:
    1. ONLY return the JSON object, no other text
    4. Summary should be 2-3 sentences describing what the program and the specific section is about.
    5. If you can't find the information, take the first few lines of the content.
    6. Never include markdown, HTML, or special characters in the values
    7. Always maintain valid JSON syntax
    8. Never include explanations or notes outside the JSON object"""
    
    try:
        # Extract program details from first chunk only
        if chunk_number == 1:
            first_lines = chunk.strip().split('\n')
            if first_lines:
                program_line = first_lines[0].strip('# ').strip()
                og_pg_name = first_lines[0].strip('# ').strip()
                
                # Normalize all dashes (en dash and em dash) to a standard hyphen (-)
                program_line = program_line.replace("\u2013", "-").replace("\u2014", "-")
                
                # Split using the standard hyphen, ensuring only two parts
                program_parts = program_line.split(" - ", 1)  # Split at the first occurrence
                
                program_name = program_parts[0].strip()
                program_mode = program_parts[1].strip() if len(program_parts) > 1 else ''
                
                # Determine program mode and campus location
                if 'Online' in program_mode:
                    mode = 'online'
                    campus_location = 'online'
                else:
                    mode = 'on campus'
                    campus_location = program_mode  # If empty, campus_location will be ''
                if not campus_location:
                    # If campus location is still empty, try to infer from the program name or elsewhere
                    # You could have some predefined list or rules for default campus locations
                    campus_location = 'Boston'

                # Create program details dictionary
                program_details = {
                    'program_name': og_pg_name.replace("\u2013", "-").replace("\u2014", "-"),
                    'program_mode': mode,
                    'campus_location': campus_location
                }
        print(program_details)
        
        # Process chunk with LLM
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": f"{system_prompt}\n\nURL: {url}\n\nContent:\n{chunk[:1000]}",
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            #extracted = json.loads(result["response"])
            # Check if 'response' key exists and is not empty
            if "response" not in result or not result["response"].strip():
                print(f"Warning: Empty response from LLM for {url}")
                extracted = {"summary": "\n".join(chunk.splitlines()[:6])}  # Fallback: First few lines
            else:
                raw_text = result["response"]

                # Attempt JSON parsing
                try:
                    extracted = json.loads(raw_text)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON received for {url}. Attempting fallback extraction.")
                    
                    # Try extracting a valid JSON object from the raw text using regex
                    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                    if json_match:
                        try:
                            extracted = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            extracted = {"summary": "\n".join(chunk.splitlines()[:3])}  # Fallback: First few lines
                    else:
                        extracted = {"summary": "\n".join(chunk.splitlines()[:3])}  # Fallback: First few lines

            # Ensure 'summary' exists in the extracted data
            summary = extracted.get("summary", "\n".join(chunk.splitlines()[:3]))
            
            # Get unique embedding for this chunk
            embedding = await get_embedding(chunk)
            
            # Create metadata with both shared and chunk-specific information
            metadata = {
                # Chunk-specific metadata
                "chunk_size": len(chunk),
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                
                # Shared metadata
                "source": "cps_program_docs",
                "url_path": urlparse(url).path
            }
            
            # Add program details if available
            if program_details:
                metadata.update(program_details)
            
            return ProgramInfo(
                url=url,
                title=metadata["program_name"],
                summary=summary,
                content=chunk,  # Unique content for each chunk
                chunk_number=chunk_number,
                embedding=embedding,  # Unique embedding for each chunk
                metadata=metadata
            )
    except Exception as e:
        print(f"Error processing chunk for {url}: {e}")
        return None

async def process_url(url: str, crawler: AsyncWebCrawler) -> List[ProgramInfo]:
    """Process a single URL and extract program information."""
    try:
        print(f"Processing: {url}")
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
            session_id="program_crawl"
        )
        
        if not result.success:
            print(f"Failed to crawl {url}: {result.error_message}")
            return []
            
        cleaned_content = process_and_modify_markdown(result.markdown_v2.raw_markdown)
        
        

        chunks = chunk_program_content(cleaned_content)
        print(len(chunks))
        # with open('program_data.jsonl', 'a') as f:
        #     f.write('\n --Partition --- \n'.join(chunks))  # Join with newlines for separate lines
        
        program_infos = []
        program_details = {}
        for i, chunk in enumerate(chunks, 1):
            info = await extract_program_info(
            chunk=chunk,
            url=url,
            chunk_number=i,
            program_details=program_details
            )
            if info:
                if i == 1:
                    # Store just the program-specific details from first chunk
                    program_details = {
                        'program_name': info.metadata['program_name'],
                        'program_mode': info.metadata['program_mode'],
                        'campus_location': info.metadata['campus_location']
                    }
                program_infos.append(info)
                await insert_chunk(info)
            
        return program_infos
        
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []

async def main():
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )
    
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        #replace the sitemap index with your college website sitemap if avaiable. 
        response = requests.get(
            "https://cps.northeastern.edu/cps-program-sitemap.xml",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        print(len(urls))
        all_program_info = []
        for url in urls:
            program_infos = await process_url(url, crawler)
            all_program_info.extend(program_infos)
            
            # Note: We're still keeping the JSONL file as backup
            with open('program_data.jsonl', 'a') as f:
                for info in program_infos:
                    f.write(json.dumps({
                        "url": info.url,
                        "title": info.title,
                        "summary": info.summary,
                        "content": info.content,
                        "chunk_number": info.chunk_number,
                        "metadata": info.metadata
                    }) + '\n')
                    
            print(f"Processed {url}: Found {len(program_infos)} chunks")
            await asyncio.sleep(2)
            
    finally:
        await crawler.close()
        
    print(f"Completed processing {len(urls)} URLs")
    print(f"Total program chunks extracted: {len(all_program_info)}")

if __name__ == "__main__":
    asyncio.run(main())
