#!/usr/bin/env python3
"""
Example script untuk re-upload ke Pinecone dengan kontrol penuh
"""

import os
from paper_search import AdvancedPaperSearcher
from dotenv import load_dotenv

load_dotenv()

def main():
    # Initialize searcher
    searcher = AdvancedPaperSearcher()
    
    # Setup Pinecone
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY not found in environment")
        return
    
    searcher.setup_pinecone(pinecone_api_key)
    
    # Load existing chunks from checkpoint
    _, chunks = searcher.load_checkpoint()
    
    if not chunks:
        print("❌ No chunks found in checkpoint. Run the main pipeline first.")
        return
    
    print(f"📊 Found {len(chunks)} chunks to re-upload")
    
    # Example 1: Re-upload semua dengan validasi dan batch kecil
    print("\n🔄 Example 1: Safe re-upload with validation")
    success = searcher.reupload_to_pinecone(
        chunks=chunks,
        force_reupload=True,      # Ignore checkpoint
        batch_size=10,            # Batch kecil untuk stability
        validate_data=True,       # Validasi data sebelum upload
        clear_index=False         # Jangan hapus index
    )
    
    if success:
        print("✅ Re-upload completed successfully!")
    else:
        print("❌ Re-upload completed with errors. Check logs.")
    
    # Example 2: Upload sebagian saja (batch 5-15)
    print("\n🔄 Example 2: Partial re-upload (batches 5-15)")
    success = searcher.reupload_to_pinecone(
        chunks=chunks,
        force_reupload=True,
        batch_size=25,
        start_batch=5,            # Mulai dari batch ke-5
        max_batches=10,           # Upload 10 batch saja
        validate_data=True
    )
    
    # Example 3: Clear index dan upload ulang semua
    print("\n🔄 Example 3: Clear index and full re-upload")
    confirm = input("⚠️  This will DELETE all data in Pinecone index. Continue? (y/N): ")
    
    if confirm.lower() == 'y':
        success = searcher.reupload_to_pinecone(
            chunks=chunks,
            force_reupload=True,
            batch_size=20,
            clear_index=True,         # HAPUS semua data di index
            validate_data=True
        )
        
        if success:
            print("✅ Full re-upload with index clearing completed!")
        else:
            print("❌ Re-upload failed. Check logs.")
    else:
        print("❌ Index clearing cancelled.")

if __name__ == "__main__":
    main()