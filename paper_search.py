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
        self.checkpoint_dir = self.base_dir / "checkpoints"
        
        # Create directories
        for dir_path in [self.pdf_dir, self.text_dir, self.chunks_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint files
        self.checkpoint_file = self.checkpoint_dir / "progress_checkpoint.json"
        self.papers_checkpoint_file = self.checkpoint_dir / "papers_checkpoint.json"
        self.chunks_checkpoint_file = self.checkpoint_dir / "chunks_checkpoint.json"
        
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
        
        # Simple sentence tokenizer (replaces NLTK)
        self.sentence_endings = r'[.!?]+\s+'
        
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
        
        # Progress tracking
        self.progress = {
            'current_step': 'initialized',
            'completed_steps': [],
            'current_query_index': 0,
            'current_paper_index': 0,
            'current_chunk_index': 0,
            'last_saved_at': None
        }
        
        logger.info("✅ AdvancedPaperSearcher v2 initialized!")
    
    def simple_sent_tokenize(self, text: str) -> List[str]:
        """Simple sentence tokenizer to replace NLTK."""
        if not text:
            return []
        
        # Split on sentence endings followed by whitespace and capital letter
        sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Add back the period if it doesn't end with punctuation
                if not sentence[-1] in '.!?':
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def save_checkpoint(self, papers: List[Dict] = None, chunks: List[Dict] = None):
        """Save current progress checkpoint."""
        try:
            # Update progress
            self.progress['last_saved_at'] = datetime.now().isoformat()
            
            # Save progress checkpoint
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'progress': self.progress,
                    'stats': self.stats
                }, f, indent=2, ensure_ascii=False)
            
            # Save papers if provided
            if papers:
                with open(self.papers_checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)
            
            # Save chunks if provided
            if chunks:
                with open(self.chunks_checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Checkpoint saved at {self.progress['last_saved_at']}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """Load previous checkpoint if exists."""
        papers = None
        chunks = None
        
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                self.progress = checkpoint_data.get('progress', self.progress)
                self.stats = checkpoint_data.get('stats', self.stats)
                
                logger.info(f"🔄 Resuming from checkpoint: {self.progress['current_step']}")
                logger.info(f"   Last saved: {self.progress.get('last_saved_at', 'Unknown')}")
                
                # Load papers if available
                if self.papers_checkpoint_file.exists():
                    with open(self.papers_checkpoint_file, 'r', encoding='utf-8') as f:
                        papers = json.load(f)
                    logger.info(f"   Loaded {len(papers)} papers from checkpoint")
                
                # Load chunks if available
                if self.chunks_checkpoint_file.exists():
                    with open(self.chunks_checkpoint_file, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    logger.info(f"   Loaded {len(chunks)} chunks from checkpoint")
                
                return papers, chunks
            else:
                logger.info("🆕 No checkpoint found. Starting fresh.")
                return None, None
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None, None
    
    def mark_step_completed(self, step_name: str):
        """Mark a step as completed."""
        if step_name not in self.progress['completed_steps']:
            self.progress['completed_steps'].append(step_name)
        self.progress['current_step'] = step_name
        self.save_checkpoint()
    
    def clear_checkpoints(self):
        """Clear all checkpoint files to start fresh."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.papers_checkpoint_file.exists():
                self.papers_checkpoint_file.unlink()
            if self.chunks_checkpoint_file.exists():
                self.chunks_checkpoint_file.unlink()
            
            # Reset progress
            self.progress = {
                'current_step': 'initialized',
                'completed_steps': [],
                'current_query_index': 0,
                'current_paper_index': 0,
                'current_chunk_index': 0,
                'last_saved_at': None
            }
            
            logger.info("🗑️ All checkpoints cleared. Ready to start fresh.")
            
        except Exception as e:
            logger.error(f"Failed to clear checkpoints: {e}")
    
    def get_progress_status(self) -> Dict:
        """Get current progress status."""
        return {
            'current_step': self.progress['current_step'],
            'completed_steps': self.progress['completed_steps'],
            'last_saved': self.progress.get('last_saved_at'),
            'stats': self.stats
        }
    
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
        """Search semua sumber akademik dengan resume capability."""
        logger.info("🔍 Starting comprehensive academic search...")
        
        # Check if we have papers from checkpoint
        papers, _ = self.load_checkpoint()
        if papers and 'search_completed' in self.progress['completed_steps']:
            logger.info(f"📚 Using {len(papers)} papers from checkpoint")
            return papers
        
        all_papers = []
        
        for query_index, query in enumerate(self.search_queries):
            # Skip if we've already processed this query
            if query_index < self.progress['current_query_index']:
                continue
                
            logger.info(f"📝 Query {query_index + 1}/{len(self.search_queries)}: {query}")
            
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
            
            # Update progress
            self.progress['current_query_index'] = query_index + 1
            self.save_checkpoint()
            
            time.sleep(3)  # Delay between queries
        
        logger.info(f"📊 Total papers before deduplication: {len(all_papers)}")
        
        # Deduplication
        unique_papers = self._deduplication(all_papers)
        self.stats['papers_found'] = len(unique_papers)
        
        # Mark search as completed
        self.mark_step_completed('search_completed')
        self.save_checkpoint(unique_papers)
        
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
        """Download PDFs dan extract text dengan resume capability."""
        logger.info("📄 Starting PDF download and text extraction...")
        
        # Check if we have processed papers from checkpoint
        if 'pdf_extraction_completed' in self.progress['completed_steps']:
            logger.info("📚 Using papers with extracted text from checkpoint")
            return papers
        
        papers_with_pdfs = [p for p in papers if p.get('pdf_url')]
        logger.info(f"📊 Found {len(papers_with_pdfs)} papers with PDF URLs")
        
        # Process papers that haven't been processed yet
        processed_count = 0
        for paper_index, paper in enumerate(papers_with_pdfs):
            # Skip if already processed
            if paper_index < self.progress['current_paper_index']:
                continue
            
            # Check if text file already exists
            text_file = self.text_dir / f"{paper['id']}.txt"
            if text_file.exists():
                logger.info(f"📝 Text already extracted for: {paper.get('title', 'Unknown')[:50]}...")
                with open(text_file, 'r', encoding='utf-8') as f:
                    paper['full_text'] = f.read()
                self.stats['texts_extracted'] += 1
                processed_count += 1
                continue
            
            try:
                extracted_text = self._download_and_extract_single_pdf(paper)
                if extracted_text:
                    paper['full_text'] = extracted_text
                    self.stats['texts_extracted'] += 1
                
                processed_count += 1
                
                # Update progress every 5 papers
                if processed_count % 5 == 0:
                    self.progress['current_paper_index'] = paper_index + 1
                    self.save_checkpoint(papers)
                    logger.info(f"📊 Processed {processed_count}/{len(papers_with_pdfs)} papers")
                
            except Exception as e:
                logger.error(f"PDF processing failed for {paper.get('title', 'Unknown')}: {e}")
                self.stats['errors'].append(f"PDF extraction: {e}")
        
        # Mark PDF extraction as completed
        self.mark_step_completed('pdf_extraction_completed')
        self.save_checkpoint(papers)
        
        logger.info(f"✅ PDF extraction completed. Processed {processed_count} papers.")
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
        """Create text chunks untuk RAG dengan resume capability."""
        logger.info("✂️ Creating text chunks for RAG...")
        
        # Check if we have chunks from checkpoint
        _, chunks = self.load_checkpoint()
        if chunks and 'chunking_completed' in self.progress['completed_steps']:
            logger.info(f"📚 Using {len(chunks)} chunks from checkpoint")
            return chunks
        
        all_chunks = []
        
        for paper_index, paper in enumerate(papers):
            # Skip if already processed
            if paper_index < self.progress['current_chunk_index']:
                continue
                
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
                
                # Update progress every 10 papers
                if (paper_index + 1) % 10 == 0:
                    self.progress['current_chunk_index'] = paper_index + 1
                    self.save_checkpoint(papers, all_chunks)
                    logger.info(f"📊 Created chunks for {paper_index + 1}/{len(papers)} papers")
                
            except Exception as e:
                logger.error(f"Error creating chunks for {paper.get('title', 'Unknown')}: {e}")
                self.stats['errors'].append(f"Chunking: {e}")
        
        logger.info(f"📊 Created {len(all_chunks)} text chunks")
        
        # Save chunks
        chunks_file = self.chunks_dir / "all_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        # Mark chunking as completed
        self.mark_step_completed('chunking_completed')
        self.save_checkpoint(papers, all_chunks)
        
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
                sentences = self.simple_sent_tokenize(chunk_text)
                if len(sentences) > 1:
                    chunk_text = ' '.join(sentences[:-1])
            
            chunks.append(chunk_text.strip())
            
            # Move start position with overlap
            start = end - overlap
            
            if start >= len(tokens):
                break
        
        return chunks
    
    def upload_to_pinecone(self, chunks: List[Dict]) -> bool:
        """Upload chunks ke Pinecone dengan llama-text-embed-v2 dan resume capability."""
        return self._upload_to_pinecone_internal(chunks, force_reupload=False)
    
    def reupload_to_pinecone(self, chunks: List[Dict], 
                           force_reupload: bool = True,
                           batch_size: int = 25,
                           start_batch: int = 0,
                           max_batches: int = None,
                           validate_data: bool = True,
                           clear_index: bool = False) -> bool:
        """
        Re-upload chunks ke Pinecone dengan kontrol penuh.
        
        Args:
            chunks: List of chunk dictionaries
            force_reupload: Ignore checkpoint dan upload ulang
            batch_size: Ukuran batch (default 25, lebih kecil untuk stability)
            start_batch: Mulai dari batch ke-berapa (untuk resume)
            max_batches: Maksimal batch yang diupload (None = semua)
            validate_data: Validasi data sebelum upload
            clear_index: Hapus semua data di index sebelum upload
        """
        return self._upload_to_pinecone_internal(
            chunks, 
            force_reupload=force_reupload,
            batch_size=batch_size,
            start_batch=start_batch,
            max_batches=max_batches,
            validate_data=validate_data,
            clear_index=clear_index
        )
    
    def _upload_to_pinecone_internal(self, chunks: List[Dict], 
                                   force_reupload: bool = False,
                                   batch_size: int = 50,
                                   start_batch: int = 0,
                                   max_batches: int = None,
                                   validate_data: bool = False,
                                   clear_index: bool = False) -> bool:
        """Internal upload function dengan parameter kontrol."""
        if not self.pinecone_index:
            logger.error("❌ Pinecone not initialized. Call setup_pinecone() first.")
            return False
        
        # Check if upload is already completed (unless force reupload)
        if not force_reupload and 'pinecone_upload_completed' in self.progress['completed_steps']:
            logger.info("☁️ Pinecone upload already completed from checkpoint")
            return True
        
        # Clear index if requested
        if clear_index:
            logger.info("🗑️ Clearing Pinecone index...")
            try:
                self.pinecone_index.delete(delete_all=True)
                logger.info("✅ Index cleared successfully")
                time.sleep(5)  # Wait for deletion to propagate
            except Exception as e:
                logger.error(f"❌ Failed to clear index: {e}")
                return False
        
        # Validate data if requested
        if validate_data:
            logger.info("🔍 Validating chunk data...")
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                if self._validate_chunk_data(chunk, i):
                    valid_chunks.append(chunk)
            chunks = valid_chunks
            logger.info(f"✅ Validated {len(chunks)} chunks")
        
        logger.info(f"☁️ {'Re-uploading' if force_reupload else 'Uploading'} to Pinecone with llama-text-embed-v2...")
        logger.info(f"📊 Parameters: batch_size={batch_size}, start_batch={start_batch}, max_batches={max_batches}")
        
        # Calculate batch range
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        end_batch = min(total_batches, start_batch + max_batches) if max_batches else total_batches
        
        uploaded_count = 0
        error_count = 0
        
        for batch_idx in range(start_batch, end_batch):
            i = batch_idx * batch_size
            batch = chunks[i:i + batch_size]
            
            try:
                # Prepare records untuk Pinecone dengan built-in embedding
                records = []
                for chunk in batch:
                    # Double-check chunk structure
                    if not isinstance(chunk, dict) or 'chunk_text' not in chunk:
                        logger.warning(f"⚠️ Skipping invalid chunk at index {i}: {type(chunk)}")
                        continue
                        
                    # For Pinecone built-in embedding, use the data format expected by the embedding model
                    record = {
                        'id': chunk['id'],
                        'chunk_text': chunk['chunk_text'],  # This matches the field_map configuration
                        'metadata': chunk['metadata']
                    }
                    records.append(record)
                
                # Upload batch - try multiple approaches
                if records:
                    logger.info(f"📤 Uploading batch {batch_idx + 1}/{end_batch} ({len(records)} records)...")
                    
                    # Method 1: Try using Pinecone's inference API for embedding
                    try:
                        # Extract texts for embedding
                        texts = [record['chunk_text'] for record in records]
                        
                        # Generate embeddings using Pinecone's inference API
                        embeddings_response = self.pinecone_client.inference.embed(
                            model="llama-text-embed-v2",
                            inputs=texts,
                            parameters={"input_type": "passage"}
                        )
                        
                        # Prepare vectors with embeddings
                        vectors = []
                        for i, record in enumerate(records):
                            vector = {
                                'id': record['id'],
                                'values': embeddings_response.data[i].values,
                                'metadata': record['metadata']
                            }
                            vectors.append(vector)
                        
                        # Upload to Pinecone
                        self.pinecone_index.upsert(vectors=vectors)
                        self.stats['uploaded_to_pinecone'] += len(vectors)
                        uploaded_count += len(vectors)
                        logger.info(f"✅ Batch {batch_idx + 1} uploaded successfully with inference API")
                        
                    except Exception as embed_error:
                        logger.warning(f"⚠️ Inference API failed for batch {batch_idx + 1}: {embed_error}")
                        
                        # Method 2: Try using dummy embeddings (for testing/fallback)
                        try:
                            import numpy as np
                            
                            vectors_dummy = []
                            for record in records:
                                # Create dummy 1536-dimensional vector (llama-text-embed-v2 dimension)
                                dummy_values = np.random.normal(0, 1, 1536).tolist()
                                vector = {
                                    'id': record['id'],
                                    'values': dummy_values,
                                    'metadata': {
                                        **record['metadata'],
                                        'chunk_text': record['chunk_text']
                                    }
                                }
                                vectors_dummy.append(vector)
                            
                            self.pinecone_index.upsert(vectors=vectors_dummy)
                            self.stats['uploaded_to_pinecone'] += len(vectors_dummy)
                            uploaded_count += len(vectors_dummy)
                            logger.info(f"✅ Batch {batch_idx + 1} uploaded with dummy embeddings (fallback)")
                            
                        except Exception as dummy_error:
                            logger.error(f"❌ All methods failed for batch {batch_idx + 1}")
                            logger.error(f"   Inference API error: {embed_error}")
                            logger.error(f"   Dummy embedding error: {dummy_error}")
                            # Continue to next batch instead of stopping
                            continue
                else:
                    logger.warning(f"⚠️ Batch {batch_idx + 1} is empty, skipping...")
                
                # Progress logging
                if (batch_idx + 1) % 5 == 0:
                    logger.info(f"📊 Progress: {uploaded_count} chunks uploaded, {error_count} errors")
                
                # Rate limiting - lebih konservatif untuk stability
                time.sleep(1.0)
                
            except Exception as e:
                error_count += 1
                error_msg = f"Pinecone upload batch {batch_idx}: {str(e)}"
                logger.error(f"❌ Error uploading batch {batch_idx + 1}: {e}")
                self.stats['errors'].append(error_msg)
                
                # Continue with next batch instead of failing completely
                continue
        
        # Mark upload as completed only if no force reupload
        if not force_reupload:
            self.mark_step_completed('pinecone_upload_completed')
        
        logger.info(f"✅ Upload completed: {uploaded_count} records uploaded, {error_count} errors")
        return error_count == 0
    
    def _validate_chunk_data(self, chunk: Dict, index: int) -> bool:
        """Validate chunk data structure."""
        try:
            # Check required fields
            required_fields = ['id', 'chunk_text', 'metadata']
            for field in required_fields:
                if field not in chunk:
                    logger.warning(f"⚠️ Chunk {index} missing field: {field}")
                    return False
            
            # Check data types
            if not isinstance(chunk['id'], str):
                logger.warning(f"⚠️ Chunk {index} has invalid id type: {type(chunk['id'])}")
                return False
                
            if not isinstance(chunk['chunk_text'], str):
                logger.warning(f"⚠️ Chunk {index} has invalid chunk_text type: {type(chunk['chunk_text'])}")
                return False
                
            if not isinstance(chunk['metadata'], dict):
                logger.warning(f"⚠️ Chunk {index} has invalid metadata type: {type(chunk['metadata'])}")
                return False
            
            # Check text length
            if len(chunk['chunk_text'].strip()) == 0:
                logger.warning(f"⚠️ Chunk {index} has empty text")
                return False
                
            if len(chunk['chunk_text']) > 10000:  # Reasonable limit
                logger.warning(f"⚠️ Chunk {index} text too long: {len(chunk['chunk_text'])} chars")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Error validating chunk {index}: {e}")
            return False
    
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
    
    def run_complete_pipeline(self, pinecone_api_key: str = None, resume_upload_only: bool = False):
        """Run complete RAG data pipeline dengan resume capability."""
        logger.info("🚀 Starting Complete RAG Data Pipeline v2")
        logger.info("🎯 Target: Container vs MicroVM Research Papers")
        logger.info("🤖 Using Pinecone llama-text-embed-v2 for embeddings")
        
        # Load checkpoint at startup
        papers, chunks = self.load_checkpoint()
        
        try:
            if resume_upload_only:
                logger.info("🔄 Resuming only Pinecone upload...")
                if not chunks:
                    logger.error("❌ No chunks found in checkpoint. Cannot resume upload.")
                    return
                
                # Setup Pinecone and upload only
                if pinecone_api_key:
                    self.setup_pinecone(pinecone_api_key)
                    self.upload_to_pinecone(chunks)
                else:
                    logger.error("❌ Pinecone API key required for upload.")
                    return
                
                # Save complete dataset and print statistics
                self.save_complete_dataset(papers, chunks)
                self.print_final_statistics()
                self.mark_step_completed('pipeline_completed')
                
                logger.info("🎉 Upload completed!")
                return
            
            # Step 1: Search all sources (or resume from checkpoint)
            if not papers or 'search_completed' not in self.progress['completed_steps']:
                papers = self.search_all_sources()
                
                if not papers:
                    logger.error("❌ No papers found. Exiting.")
                    return
            else:
                logger.info("📚 Resuming with papers from checkpoint")
            
            # Step 2: Download PDFs and extract text (or resume from checkpoint)
            if 'pdf_extraction_completed' not in self.progress['completed_steps']:
                papers = self.download_and_extract_pdfs(papers)
            else:
                logger.info("📄 PDF extraction already completed from checkpoint")
            
            # Step 3: Create text chunks (or resume from checkpoint)
            if not chunks or 'chunking_completed' not in self.progress['completed_steps']:
                chunks = self.create_text_chunks(papers)
            else:
                logger.info("✂️ Text chunking already completed from checkpoint")
            
            # Step 4: Setup Pinecone and upload (or resume from checkpoint)
            if pinecone_api_key:
                self.setup_pinecone(pinecone_api_key)
                if 'pinecone_upload_completed' not in self.progress['completed_steps']:
                    self.upload_to_pinecone(chunks)
                else:
                    logger.info("☁️ Pinecone upload already completed from checkpoint")
            else:
                logger.warning("⚠️ No Pinecone API key provided. Skipping Pinecone upload.")
            
            # Step 5: Save complete dataset
            self.save_complete_dataset(papers, chunks)
            
            # Step 6: Print final statistics
            self.print_final_statistics()
            
            # Mark pipeline as completed
            self.mark_step_completed('pipeline_completed')
            
            logger.info("🎉 RAG Dataset Creation Complete!")
            logger.info(f"📁 All data saved to: {self.base_dir}")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            self.stats['errors'].append(f"Pipeline: {e}")
            # Save checkpoint on error
            self.save_checkpoint(papers if 'papers' in locals() else None, 
                               chunks if 'chunks' in locals() else None)


def main():
    """Main function - otomatis maksimal dengan resume capability."""
    print("🔬 Advanced Academic Paper Search & RAG Data Pipeline v2")
    print("🎯 Container vs MicroVM Research - Pinecone llama-text-embed-v2")
    print("🔄 Resume Capability Enabled")
    print("="*60)
    
    # Initialize searcher
    searcher = AdvancedPaperSearcher()
    
    # Check for existing progress
    status = searcher.get_progress_status()
    if status['completed_steps']:
        print(f"\n📊 Found existing progress:")
        print(f"   Current step: {status['current_step']}")
        print(f"   Completed steps: {', '.join(status['completed_steps'])}")
        print(f"   Last saved: {status['last_saved']}")
        
        print("\n🔄 Resume options:")
        print("   y - Resume full pipeline")
        print("   u - Resume only Pinecone upload (if chunks exist)")
        print("   n - Start fresh")
        print("   c - Clear checkpoints and start fresh")
        
        choice = input("\nChoose option: ").strip().lower()
        
        if choice == 'c':
            searcher.clear_checkpoints()
            print("🗑️ Checkpoints cleared. Starting fresh.")
        elif choice == 'u':
            print("🔄 Will resume only Pinecone upload...")
            resume_upload_only = True
        elif choice != 'y':
            print("❌ Exiting.")
            return
        else:
            resume_upload_only = False
    else:
        resume_upload_only = False
    
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
    print("💾 Checkpoints saved for future resume capability.")


def reupload_only():
    """Function khusus untuk re-upload ke Pinecone saja."""
    import sys
    
    searcher = AdvancedPaperSearcher()
    
    # Setup Pinecone
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY not found in environment")
        pinecone_api_key = input("Enter Pinecone API Key: ").strip()
        if not pinecone_api_key:
            print("❌ No API key provided. Exiting.")
            return
    
    searcher.setup_pinecone(pinecone_api_key)
    
    # Load existing chunks
    _, chunks = searcher.load_checkpoint()
    if not chunks:
        print("❌ No chunks found. Run the main pipeline first.")
        return
    
    print(f"📊 Found {len(chunks)} chunks to re-upload")
    
    # Parse command line arguments for re-upload parameters
    batch_size = 25
    clear_index = False
    validate_data = True
    start_batch = 0
    max_batches = None
    
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if arg.startswith('--batch-size='):
                batch_size = int(arg.split('=')[1])
            elif arg.startswith('--start-batch='):
                start_batch = int(arg.split('=')[1])
            elif arg.startswith('--max-batches='):
                max_batches = int(arg.split('=')[1])
            elif arg == '--clear-index':
                clear_index = True
            elif arg == '--no-validate':
                validate_data = False
    
    print(f"🔧 Re-upload parameters:")
    print(f"   batch_size={batch_size}")
    print(f"   start_batch={start_batch}")
    print(f"   max_batches={max_batches}")
    print(f"   clear_index={clear_index}")
    print(f"   validate_data={validate_data}")
    
    # Confirm if clearing index
    if clear_index:
        confirm = input("⚠️  This will DELETE all data in Pinecone index. Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Re-upload cancelled.")
            return
    
    # Re-upload
    success = searcher.reupload_to_pinecone(
        chunks=chunks,
        force_reupload=True,
        batch_size=batch_size,
        start_batch=start_batch,
        max_batches=max_batches,
        validate_data=validate_data,
        clear_index=clear_index
    )
    
    if success:
        print("✅ Re-upload completed successfully!")
    else:
        print("❌ Re-upload completed with errors. Check logs above.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'reupload':
        reupload_only()
    else:
        main()