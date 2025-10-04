#!/usr/bin/env python3
"""
NEOSSAT Data Downloader
Downloads files from Canadian Space Agency's NEOSSAT ASTRO data repository
"""

import os
import requests
import re
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
from typing import List, Optional
from tqdm import tqdm

class NEOSSATDownloader:
    def __init__(self, base_url: str = "https://data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/NEOSSAT/ASTRO/2025/"):
        self.base_url = base_url
        self.data_dir = Path("data")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)
    
    def _format_bytes(self, bytes_value: int) -> str:
        """
        Format bytes into human readable format
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
    
    def get_file_links(self) -> List[str]:
        """
        Parse the directory listing and extract file links
        """
        try:
            print(f"Fetching directory listing from: {self.base_url}")
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML to find file links
            # Look for href attributes that point to files (not directories)
            file_pattern = r'href="([^"]+\.[a-zA-Z0-9]+)"'
            matches = re.findall(file_pattern, response.text)
            
            # Filter out parent directory links and navigation links
            file_links = []
            for match in matches:
                if not match.startswith('..') and not match.startswith('/') and not match.startswith('?'):
                    file_links.append(match)
            
            print(f"Found {len(file_links)} files to download")
            return file_links
            
        except requests.RequestException as e:
            print(f"Error fetching directory listing: {e}")
            return []
    
    def download_file(self, filename: str, pbar_overall: tqdm = None) -> bool:
        """
        Download a single file with minimal output
        """
        file_url = urljoin(self.base_url, filename)
        local_path = self.data_dir / filename
        
        # Skip if file already exists
        if local_path.exists():
            return True
        
        try:
            response = self.session.get(file_url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except requests.RequestException as e:
            # Remove partial file if it exists
            if local_path.exists():
                local_path.unlink()
            return False
        except Exception as e:
            if local_path.exists():
                local_path.unlink()
            return False
    
    def download_all(self) -> None:
        """
        Download all files from the repository
        """
        print("🚀 NEOSSAT Data Downloader")
        print("=" * 60)
        
        file_links = self.get_file_links()
        if not file_links:
            print("❌ No files found to download")
            return
        
        print(f"\n📂 Target directory: {self.data_dir.absolute()}")
        print(f"📊 Found {len(file_links)} files to download")
        print("-" * 60)
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        start_time = time.time()
        
        # Overall progress bar - sadece toplam ilerleme
        with tqdm(
            total=len(file_links),
            desc="📥 Downloading files",
            unit="file",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] [{elapsed}<{remaining}]"
        ) as pbar:
            
            for filename in file_links:
                # Dosya zaten varsa kontrol et
                local_path = self.data_dir / filename
                if local_path.exists():
                    skipped_count += 1
                elif self.download_file(filename):
                    success_count += 1
                else:
                    failed_count += 1
                
                # Progress bar'ı güncelle - dosya adını postfix olarak göster
                pbar.set_postfix_str(f"Current: {filename[:30]}..." if len(filename) > 30 else filename)
                pbar.update(1)
                
                # Small delay to be respectful to the server
                time.sleep(0.2)
        
        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"🎉 Download completed in {elapsed_time:.1f} seconds")
        print(f"✅ Successfully downloaded: {success_count} files")
        if skipped_count > 0:
            print(f"📁 Skipped (already exists): {skipped_count} files")
        if failed_count > 0:
            print(f"❌ Failed downloads: {failed_count} files")
        print(f"📁 Files saved to: {self.data_dir.absolute()}")


def main():
    """
    Main function to run the downloader
    """
    # You can customize the URL here if needed
    url = "https://data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/NEOSSAT/ASTRO/2025/001/"
    
    downloader = NEOSSATDownloader(url)
    
    try:
        downloader.download_all()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()