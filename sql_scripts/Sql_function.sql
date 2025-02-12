-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(768),  -- nomic-embed-text embeddings are 768 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
create index on site_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_site_pages_metadata on site_pages using gin (metadata);


create or replace function match_site_pages (
  query_embedding vector(768),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  search_mode varchar default 'general'
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
declare
  top_program_title varchar;
begin
 if search_mode = 'general' then
   -- For general search, retrieve top 5 most relevant chunks across all programs
   return query
   select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
   from site_pages
   where metadata @> filter
   order by similarity desc
   limit 5;
 else
  -- Step 1: Find the most relevant program based on the title and metadata
  select title into top_program_title
  from site_pages
  where metadata @> filter
  order by 1 - (site_pages.embedding <=> query_embedding) desc
  limit 1;

  -- Step 2: Retrieve all three chunks for the most relevant program
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where title = top_program_title and metadata @> filter
  order by chunk_number
  limit 3;
  end if;
end;
$$;

alter table site_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on site_pages
  for select
  to public
  using (true);
