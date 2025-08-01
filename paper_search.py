#!/usr/bin/env python3
"""
Advanced Academic Paper Search & RAG Data Pipeline v2
Optimized untuk Pinecone llama-text-embed-v2
"""

import requests
import json
import time
import os
import re
import hashlib
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from datetime import datetime
import concurrent.futures
from pathlib import Path
import logging

# PDF processing
import PyPDF2
import fitz  # PyMuPDF
from io import BytesIO

# Text processing
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize

# Pinecone
from pinecone import Pinecone

# Data processing
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdvancedPaperSearcher:
    def __init__(self):
        """Initialize dengan konfigurasi maksimal untuk RAG AI."""
        
        # Directories
        self.base_dir = Path("rag_data")
        self.pdf_dir = self.base_dir / "pdfs"
        self.text_dir = self.base_dir / "extracted_texts"
        self.chunks_dir = self.base_dir / "text_chunks"
        
        # Create directories
        for dir_path in [self.pdf_dir, self.text_dir, self.chunks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # API endpoints
        self.apis = {
            'semantic_scholar': "https://api.semanticscholar.org/graph/v1",
            'arxiv': "http://export.arxiv.org/api/query",
            'crossref': "https://api.crossref.org/works",
            'pubmed': "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            'core': "https://api.core.ac.uk/v3",
            'openalex': "https://api.openalex.org/works",
            'ieee': "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        }
        
        # Search queries - comprehensive untuk containers vs microVMs
        self.search_queries = [
            "containers microVMs isolation security Firecracker Kata",
            "container security isolation microVM virtualization",
            "Firecracker security isolation performance",
            "Kata Containers security isolation performance",
            "gVisor security isolation containers",
            "microVM container security comparison",
            "lightweight virtualization security containers",
            "Docker security isolation performance benchmark",
            "container escape vulnerability security",
            "microVM security boundary isolation",
            "container runtime security comparison",
            "container performance overhead isolation",
            "microVM performance benchmark comparison",
            "serverless containers security isolation",
            "edge computing containers microVMs",
            "cloud native security containers"
        ]
        
        # Initialize tokenizer untuk chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("📚 Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
        
        # Pinecone setup
        self.pinecone_client = None
        self.pinecone_index = None
        
        # Statistics tracking
        self.stats = {
            'papers_found': 0,
            'pdfs_downloaded': 0,
            'texts_extracted': 0,
            'chunks_created': 0,
            'uploaded_to_pinecone': 0,
            'errors': []
        }
        
        logger.info("✅ AdvancedPaperSearcher v2 initialized!")
    
    def setup_pinecone(self, api_key: str, index_name: str = "container-microvm-rag"):
        """Setup Pinecone dengan llama-text-embed-v2."""
        try:
            logger.info("🔧 Setting up Pinecone with llama-text-embed-v2...")
            
            # Initialize Pinecone
            self.pinecone_client = Pinecone(api_key=api_key)
            
            # Create index if not exists
            if not self.pinecone_client.has_index(index_name):
                logger.info(f"📊 Creating Pinecone index: {index_name}")
                self.pinecone_client.create_index_for_model(
                    name=index_name,
                    cloud="aws",
                    region="us-east-1",
                    embed={
                        "model": "llama-text-embed-v2",
                        "field_map": {"text": "chunk_text"}
                    }
                )
                time.sleep(20)  # Wait for index creation
            
            self.pinecone_index = self.pinecone_client.Index(index_name)
            logger.info("✅ Pinecone setup completed with llama-text-embed-v2!")
            
        except Exception as e:
            logger.error(f"❌ Pinecone setup failed: {e}")
            self.stats['errors'].append(f"Pinecone setup: {e}")
    
    def search_all_sources(self) -> List[Dict]:
        """Search semua sumber akademik."""
        logger.info("🔍 Starting comprehensive academic search...")
        all_papers = []
        
        for query in self.search_queries:
            logger.info(f"📝 Query: {query}")
            
            # Search each source
            sources = [
                ('arXiv', self._search_arxiv),
                ('Semantic Scholar', self._search_semantic_scholar),
                ('CrossRef', self._search_crossref),
                ('OpenAlex', self._search_openalex)
            ]
            
            for source_name, search_func in sources:
                try:
                    papers = search_func(query, max_results=30)
                    all_papers.extend(papers)
                    logger.info(f"   {source_name}: {len(papers)} papers")
                    time.sleep(2)  # Rate limiting
                except Exception as e:
                    logger.error(f"   {source_name} error: {e}")
                    self.stats['errors'].append(f"{source_name}: {e}")
            
            time.sleep(3)  # Delay between queries
        
        logger.info(f"📊 Total papers before deduplication: {len(all_papers)}")
        
        # Deduplication
        unique_papers = self._deduplication(all_papers)
        self.stats['papers_found'] = len(unique_papers)
        
        logger.info(f"📊 Unique papers after deduplication: {len(unique_papers)}")
        return unique_papers
    
    def _search_arxiv(self, query: str, max_results: int = 30) -> List[Dict]:
        """Search arXiv."""
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.apis['arxiv'], params=params, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            papers = []
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper = self._parse_arxiv_entry(entry, ns)
                if paper:
                    papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []
    
    def _parse_arxiv_entry(self, entry, ns) -> Optional[Dict]:
        """Parse arXiv entry."""
        try:
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            
            # Authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            # Abstract
            summary = entry.find('atom:summary', ns)
            abstract = summary.text.strip().replace('\n', ' ') if summary is not None else ""
            
            # Date
            published = entry.find('atom:published', ns)
            pub_date = published.text if published is not None else ""
            pub_year = pub_date[:4] if pub_date else "Unknown"
            
            # Links
            arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            paper_url = f"https://arxiv.org/abs/{arxiv_id}"
            
            return {
                'id': f"arxiv_{arxiv_id}",
                'source': 'arXiv',
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'year': pub_year,
                'url': paper_url,
                'pdf_url': pdf_url,
                'venue': 'arXiv preprint',
                'citation_count': 0
            }
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {e}")
            return None
    
    def _search_semantic_scholar(self, query: str, max_results: int = 30) -> List[Dict]:
        """Search Semantic Scholar."""
        url = f"{self.apis['semantic_scholar']}/paper/search"
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'title,authors,abstract,year,url,openAccessPdf,citationCount,venue'
        }
        
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    wait_time = (2 ** attempt) * 10
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                papers = []
                
                for paper_data in data.get('data', []):
                    paper = self._parse_semantic_scholar_entry(paper_data)
                    if paper:
                        papers.append(paper)
                
                return papers
                
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Semantic Scholar search failed: {e}")
                    return []
                time.sleep(5)
        
        return []
    
    def _parse_semantic_scholar_entry(self, data) -> Optional[Dict]:
        """Parse Semantic Scholar entry."""
        try:
            title = data.get('title', 'No title')
            abstract = data.get('abstract', '')
            year = str(data.get('year', 'Unknown'))
            url = data.get('url', '')
            
            # Authors
            authors = []
            for author in data.get('authors', []):
                if author.get('name'):
                    authors.append(author['name'])
            
            # PDF URL
            pdf_url = None
            open_access = data.get('openAccessPdf')
            if open_access and open_access.get('url'):
                pdf_url = open_access['url']
            
            citation_count = data.get('citationCount', 0)
            venue = data.get('venue', '')
            
            paper_id = f"ss_{hashlib.md5(title.encode()).hexdigest()[:12]}"
            
            return {
                'id': paper_id,
                'source': 'Semantic Scholar',
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'year': year,
                'url': url,
                'pdf_url': pdf_url,
                'venue': venue,
                'citation_count': citation_count
            }
        except Exception as e:
            logger.error(f"Error parsing Semantic Scholar entry: {e}")
            return None   
 
    def _search_crossref(self, query: str, max_results: int = 30) -> List[Dict]:
        """Search CrossRef."""
        params = {
            'query': query,
            'rows': max_results,
            'sort': 'relevance',
            'order': 'desc'
        }
        
        try:
            response = requests.get(self.apis['crossref'], params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('message', {}).get('items', []):
                paper = self._parse_crossref_entry(item)
                if paper:
                    papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"CrossRef search error: {e}")
            return []
    
    def _parse_crossref_entry(self, item) -> Optional[Dict]:
        """Parse CrossRef entry."""
        try:
            title_list = item.get('title', [])
            title = title_list[0] if title_list else 'No title'
            
            # Authors
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                full_name = f"{given} {family}".strip()
                if full_name:
                    authors.append(full_name)
            
            # Abstract
            abstract_list = item.get('abstract', [])
            abstract = abstract_list[0] if abstract_list else ''
            
            # Year
            published = item.get('published-print') or item.get('published-online')
            year = 'Unknown'
            if published and 'date-parts' in published:
                date_parts = published['date-parts'][0]
                if date_parts:
                    year = str(date_parts[0])
            
            # DOI and URL
            doi = item.get('DOI', '')
            url = f"https://doi.org/{doi}" if doi else item.get('URL', '')
            
            # Venue
            venue = ''
            container_title = item.get('container-title', [])
            if container_title:
                venue = container_title[0]
            
            citation_count = item.get('is-referenced-by-count', 0)
            
            paper_id = f"crossref_{hashlib.md5(title.encode()).hexdigest()[:12]}"
            
            return {
                'id': paper_id,
                'source': 'CrossRef',
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'year': year,
                'url': url,
                'pdf_url': None,
                'venue': venue,
                'citation_count': citation_count
            }
        except Exception as e:
            logger.error(f"Error parsing CrossRef entry: {e}")
            return None
    
    def _search_openalex(self, query: str, max_results: int = 30) -> List[Dict]:
        """Search OpenAlex."""
        params = {
            'search': query,
            'per-page': max_results,
            'sort': 'relevance_score:desc'
        }
        
        try:
            response = requests.get(self.apis['openalex'], params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for work in data.get('results', []):
                paper = self._parse_openalex_entry(work)
                if paper:
                    papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"OpenAlex search error: {e}")
            return []
    
    def _parse_openalex_entry(self, work) -> Optional[Dict]:
        """Parse OpenAlex entry."""
        try:
            title = work.get('title', 'No title')
            abstract = work.get('abstract', '')
            
            # Authors
            authors = []
            for authorship in work.get('authorships', []):
                author = authorship.get('author', {})
                display_name = author.get('display_name', '')
                if display_name:
                    authors.append(display_name)
            
            # Year
            publication_year = work.get('publication_year')
            year = str(publication_year) if publication_year else 'Unknown'
            
            # DOI and URL
            doi = work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else ''
            url = work.get('doi', '') or f"https://openalex.org/{work.get('id', '').split('/')[-1]}"
            
            # Venue
            venue = ''
            primary_location = work.get('primary_location', {})
            if primary_location:
                source = primary_location.get('source', {})
                venue = source.get('display_name', '') if source else ''
            
            # Citation count
            citation_count = work.get('cited_by_count', 0)
            
            # PDF URL
            pdf_url = None
            if primary_location and primary_location.get('pdf_url'):
                pdf_url = primary_location['pdf_url']
            
            paper_id = f"openalex_{work.get('id', '').split('/')[-1]}"
            
            return {
                'id': paper_id,
                'source': 'OpenAlex',
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'year': year,
                'url': url,
                'pdf_url': pdf_url,
                'venue': venue,
                'citation_count': citation_count
            }
        except Exception as e:
            logger.error(f"Error parsing OpenAlex entry: {e}")
            return None
    
    def _deduplication(self, papers: List[Dict]) -> List[Dict]:
        """Simple deduplication berdasarkan title similarity."""
        logger.info("🔄 Performing deduplication...")
        
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Normalize title
            title = paper.get('title', '').lower().strip()
            title_normalized = re.sub(r'[^\w\s]', '', title)
            
            # Check similarity dengan titles yang sudah ada
            is_duplicate = False
            for seen_title in seen_titles:
                if self._calculate_similarity(title_normalized, seen_title) > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(title_normalized)
        
        return unique_papers
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def download_and_extract_pdfs(self, papers: List[Dict]) -> List[Dict]:
        """Download PDFs dan extract text."""
        logger.info("📄 Starting PDF download and text extraction...")
        
        papers_with_pdfs = [p for p in papers if p.get('pdf_url')]
        logger.info(f"📊 Found {len(papers_with_pdfs)} papers with PDF URLs")
        
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_paper = {
                executor.submit(self._download_and_extract_single_pdf, paper): paper 
                for paper in papers_with_pdfs
            }
            
            for future in concurrent.futures.as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    extracted_text = future.result()
                    if extracted_text:
                        paper['full_text'] = extracted_text
                        self.stats['texts_extracted'] += 1
                except Exception as e:
                    logger.error(f"PDF processing failed for {paper.get('title', 'Unknown')}: {e}")
                    self.stats['errors'].append(f"PDF extraction: {e}")
        
        return papers
    
    def _download_and_extract_single_pdf(self, paper: Dict) -> Optional[str]:
        """Download dan extract text dari single PDF."""
        pdf_url = paper.get('pdf_url')
        if not pdf_url:
            return None
        
        try:
            # Generate safe filename
            safe_title = re.sub(r'[^\w\s-]', '', paper.get('title', 'unknown'))[:50]
            filename = f"{paper['id']}_{safe_title}.pdf"
            filepath = self.pdf_dir / filename
            
            # Download PDF
            logger.info(f"⬇️ Downloading: {paper.get('title', 'Unknown')[:50]}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()
            
            # Save PDF
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.stats['pdfs_downloaded'] += 1
            
            # Extract text
            extracted_text = self._extract_text_from_pdf(filepath)
            
            # Save extracted text
            if extracted_text:
                text_file = self.text_dir / f"{paper['id']}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_url}: {e}")
            return None
    
    def _extract_text_from_pdf(self, filepath: Path) -> Optional[str]:
        """Extract text dari PDF."""
        try:
            # Method 1: PyMuPDF (fitz)
            try:
                doc = fitz.open(filepath)
                text_parts = []
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(text)
                
                doc.close()
                
                if text_parts:
                    full_text = '\n\n'.join(text_parts)
                    # Clean text
                    full_text = re.sub(r'\n+', '\n', full_text)
                    full_text = re.sub(r'\s+', ' ', full_text)
                    return full_text.strip()
                    
            except Exception as e:
                logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")
            
            # Method 2: PyPDF2 fallback
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(text)
                
                if text_parts:
                    full_text = '\n\n'.join(text_parts)
                    full_text = re.sub(r'\n+', '\n', full_text)
                    full_text = re.sub(r'\s+', ' ', full_text)
                    return full_text.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return None
    
    def create_text_chunks(self, papers: List[Dict]) -> List[Dict]:
        """Create text chunks untuk RAG."""
        logger.info("✂️ Creating text chunks for RAG...")
        
        all_chunks = []
        
        for paper in papers:
            try:
                # Combine all text sources
                text_sources = []
                
                if paper.get('title'):
                    text_sources.append(f"Title: {paper['title']}")
                
                if paper.get('abstract'):
                    text_sources.append(f"Abstract: {paper['abstract']}")
                
                if paper.get('full_text'):
                    text_sources.append(f"Full Text: {paper['full_text']}")
                
                combined_text = '\n\n'.join(text_sources)
                
                # Create chunks
                chunks = self._split_text_into_chunks(combined_text, max_tokens=512, overlap=50)
                
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        'id': f"{paper['id']}_chunk_{i}",
                        'paper_id': paper['id'],
                        'chunk_index': i,
                        'chunk_text': chunk,  # Field untuk Pinecone embedding
                        'metadata': {
                            'title': paper.get('title', ''),
                            'authors': ', '.join(paper.get('authors', [])),
                            'year': paper.get('year', ''),
                            'source': paper.get('source', ''),
                            'venue': paper.get('venue', ''),
                            'citation_count': paper.get('citation_count', 0),
                            'url': paper.get('url', ''),
                            'paper_id': paper['id']
                        }
                    }
                    all_chunks.append(chunk_data)
                
                self.stats['chunks_created'] += len(chunks)
                
            except Exception as e:
                logger.error(f"Error creating chunks for {paper.get('title', 'Unknown')}: {e}")
                self.stats['errors'].append(f"Chunking: {e}")
        
        logger.info(f"📊 Created {len(all_chunks)} text chunks")
        
        # Save chunks
        chunks_file = self.chunks_dir / "all_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        return all_chunks
    
    def _split_text_into_chunks(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        """Split text into chunks dengan token limit."""
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= max_tokens:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Try to break at sentence boundary
            if end < len(tokens):
                sentences = sent_tokenize(chunk_text)
                if len(sentences) > 1:
                    chunk_text = ' '.join(sentences[:-1])
            
            chunks.append(chunk_text.strip())
            
            # Move start position with overlap
            start = end - overlap
            
            if start >= len(tokens):
                break
        
        return chunks
    
    def upload_to_pinecone(self, chunks: List[Dict]) -> bool:
        """Upload chunks ke Pinecone dengan llama-text-embed-v2."""
        if not self.pinecone_index:
            logger.error("❌ Pinecone not initialized. Call setup_pinecone() first.")
            return False
        
        logger.info("☁️ Uploading to Pinecone with llama-text-embed-v2...")
        
        # Batch upload
        batch_size = 50
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                # Prepare records untuk Pinecone
                records = []
                for chunk in batch:
                    record = {
                        'id': chunk['id'],
                        'text': chunk['chunk_text'],  # Field yang akan di-embed oleh llama-text-embed-v2
                        'metadata': chunk['metadata']
                    }
                    records.append(record)
                
                # Upload batch
                if records:
                    self.pinecone_index.upsert(vectors=records)
                    self.stats['uploaded_to_pinecone'] += len(records)
                
                # Progress logging
                logger.info(f"   Uploaded batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error uploading batch {i//batch_size}: {e}")
                self.stats['errors'].append(f"Pinecone upload batch {i//batch_size}: {e}")
        
        logger.info(f"✅ Uploaded {self.stats['uploaded_to_pinecone']} records to Pinecone")
        return True
    
    def save_complete_dataset(self, papers: List[Dict], chunks: List[Dict]):
        """Save complete dataset."""
        logger.info("💾 Saving complete dataset...")
        
        # Create comprehensive dataset
        dataset = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_papers': len(papers),
                'total_chunks': len(chunks),
                'sources': list(set(p['source'] for p in papers)),
                'statistics': self.stats
            },
            'papers': papers,
            'chunks': chunks
        }
        
        # Save main dataset
        dataset_file = self.base_dir / "complete_rag_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # Save papers metadata
        papers_file = self.base_dir / "papers_metadata.json"
        with open(papers_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        # Save training format
        training_data = []
        for chunk in chunks:
            training_item = {
                'text': chunk['chunk_text'],
                'metadata': chunk['metadata']
            }
            training_data.append(training_item)
        
        training_file = self.base_dir / "training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        # Create CSV
        df_papers = pd.DataFrame(papers)
        df_papers.to_csv(self.base_dir / "papers_analysis.csv", index=False)
        
        logger.info(f"✅ Complete dataset saved to {self.base_dir}")
    
    def print_final_statistics(self):
        """Print comprehensive statistics."""
        logger.info("\n" + "="*60)
        logger.info("📊 FINAL STATISTICS - RAG DATASET CREATION")
        logger.info("="*60)
        
        logger.info(f"📚 Papers found: {self.stats['papers_found']}")
        logger.info(f"📄 PDFs downloaded: {self.stats['pdfs_downloaded']}")
        logger.info(f"📝 Texts extracted: {self.stats['texts_extracted']}")
        logger.info(f"✂️ Chunks created: {self.stats['chunks_created']}")
        logger.info(f"☁️ Uploaded to Pinecone: {self.stats['uploaded_to_pinecone']}")
        
        if self.stats['errors']:
            logger.info(f"⚠️ Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:
                logger.info(f"   - {error}")
        
        logger.info("="*60)
    
    def run_complete_pipeline(self, pinecone_api_key: str = None):
        """Run complete RAG data pipeline."""
        logger.info("🚀 Starting Complete RAG Data Pipeline v2")
        logger.info("🎯 Target: Container vs MicroVM Research Papers")
        logger.info("🤖 Using Pinecone llama-text-embed-v2 for embeddings")
        
        try:
            # Step 1: Search all sources
            papers = self.search_all_sources()
            
            if not papers:
                logger.error("❌ No papers found. Exiting.")
                return
            
            # Step 2: Download PDFs and extract text
            papers = self.download_and_extract_pdfs(papers)
            
            # Step 3: Create text chunks
            chunks = self.create_text_chunks(papers)
            
            # Step 4: Setup Pinecone and upload
            if pinecone_api_key:
                self.setup_pinecone(pinecone_api_key)
                self.upload_to_pinecone(chunks)
            else:
                logger.warning("⚠️ No Pinecone API key provided. Skipping Pinecone upload.")
            
            # Step 5: Save complete dataset
            self.save_complete_dataset(papers, chunks)
            
            # Step 6: Print final statistics
            self.print_final_statistics()
            
            logger.info("🎉 RAG Dataset Creation Complete!")
            logger.info(f"📁 All data saved to: {self.base_dir}")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            self.stats['errors'].append(f"Pipeline: {e}")


def main():
    """Main function - otomatis maksimal."""
    print("🔬 Advanced Academic Paper Search & RAG Data Pipeline v2")
    print("🎯 Container vs MicroVM Research - Pinecone llama-text-embed-v2")
    print("="*60)
    
    # Initialize searcher
    searcher = AdvancedPaperSearcher()
    
    # Get Pinecone API key
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    
    if not pinecone_api_key:
        print("\n💡 Enter Pinecone API key for RAG storage:")
        pinecone_api_key = input("Pinecone API Key: ").strip()
        if not pinecone_api_key:
            print("⚠️ No API key provided. Will skip Pinecone upload.")
            pinecone_api_key = None
    
    # Run complete pipeline
    searcher.run_complete_pipeline(pinecone_api_key)
    
    print(f"\n✅ Process completed! Check the '{searcher.base_dir}' directory for all results.")
    print("🎯 Your RAG dataset is ready for AI training and semantic search!")


if __name__ == "__main__":
    main()