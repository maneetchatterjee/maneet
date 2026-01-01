#!/usr/bin/env python3
"""
Web scraper for GE Healthcare Patient Monitoring website.
This script scrapes the specified URL and generates a summary of the content.

Usage:
    python scrape_gehealthcare.py              # Scrape the actual website
    python scrape_gehealthcare.py --demo       # Run with demo/mock data
    python scrape_gehealthcare.py --help       # Show help
"""

import requests
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Any
import sys
import argparse


def get_demo_data() -> Dict[str, Any]:
    """
    Return mock data for demonstration purposes.
    This simulates what would be scraped from the GE Healthcare website.
    
    Returns:
        Dictionary containing mock scraped data
    """
    return {
        'url': 'https://www.gehealthcare.in/products/patient-monitoring',
        'title': 'Patient Monitoring | GE HealthCare India',
        'description': 'Explore GE HealthCare\'s range of patient monitoring solutions designed to help clinicians make informed decisions for better patient outcomes.',
        'headings': [
            {'level': 'h1', 'text': 'Patient Monitoring Solutions'},
            {'level': 'h2', 'text': 'Comprehensive Patient Monitoring'},
            {'level': 'h2', 'text': 'Vital Signs Monitoring'},
            {'level': 'h2', 'text': 'Multi-Parameter Monitors'},
            {'level': 'h2', 'text': 'Anesthesia Monitoring'},
            {'level': 'h2', 'text': 'Central Monitoring Systems'},
            {'level': 'h2', 'text': 'Connectivity Solutions'},
            {'level': 'h3', 'text': 'Advanced Analytics'},
            {'level': 'h3', 'text': 'Clinical Decision Support'},
            {'level': 'h3', 'text': 'Integrated Care Solutions'},
        ],
        'paragraphs': [
            'GE HealthCare offers a comprehensive portfolio of patient monitoring solutions that help clinicians make informed decisions for improved patient outcomes. Our monitoring systems are designed to provide accurate, reliable data in critical care environments.',
            'Our patient monitoring solutions include multi-parameter monitors, vital signs monitors, anesthesia monitors, and central monitoring systems. These devices are engineered to deliver clinical excellence while enhancing workflow efficiency.',
            'With advanced connectivity features, our monitoring solutions integrate seamlessly with hospital information systems, enabling better care coordination and data management. The systems support real-time data transmission and remote monitoring capabilities.',
            'GE HealthCare\'s patient monitoring technology incorporates advanced algorithms for early warning scores, alarm management, and clinical decision support. These features help reduce alarm fatigue and improve patient safety.',
            'Our monitoring solutions are scalable and flexible, suitable for various clinical settings including intensive care units, operating rooms, emergency departments, and general wards.',
        ],
        'links_count': 45,
        'products': [
            {'name': 'CARESCAPE B850 Patient Monitor'},
            {'name': 'CARESCAPE B650 Patient Monitor'},
            {'name': 'CARESCAPE B450 Patient Monitor'},
            {'name': 'Dash Series Patient Monitors'},
            {'name': 'ApexPro Series Telemetry'},
            {'name': 'CARESCAPE Gateway'},
            {'name': 'CARESCAPE Central Station'},
            {'name': 'Vital Signs Monitors'},
            {'name': 'Anesthesia Monitoring Solutions'},
            {'name': 'Neonatal Monitoring'},
        ],
        'status': 'success',
        'note': 'This is demo data for testing purposes'
    }


def scrape_website(url: str) -> Dict[str, Any]:
    """
    Scrape the website and extract relevant information.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Dictionary containing scraped data
    """
    print(f"Scraping website: {url}")
    
    try:
        # Send GET request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract page title
        title = soup.find('title')
        page_title = title.get_text().strip() if title else "No title found"
        
        # Extract main headings
        headings = []
        for heading_tag in ['h1', 'h2', 'h3']:
            for heading in soup.find_all(heading_tag):
                text = heading.get_text().strip()
                if text and len(text) > 0:
                    headings.append({
                        'level': heading_tag,
                        'text': text
                    })
        
        # Extract paragraphs
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if text and len(text) > 20:  # Filter out very short paragraphs
                paragraphs.append(text)
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '').strip() if meta_desc else ""
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            link_text = link.get_text().strip()
            if link_text and href:
                links.append({
                    'text': link_text,
                    'href': href
                })
        
        # Try to find product information
        products = []
        # Look for common product-related class names
        product_containers = soup.find_all(['div', 'section'], 
                                          class_=lambda x: x and any(keyword in str(x).lower() 
                                          for keyword in ['product', 'card', 'item']))
        
        for container in product_containers[:20]:  # Limit to first 20 to avoid noise
            product_name = container.find(['h2', 'h3', 'h4', 'p'], 
                                         class_=lambda x: x and ('title' in str(x).lower() or 
                                         'name' in str(x).lower()))
            if product_name:
                text = product_name.get_text().strip()
                if text and text not in [p.get('name', '') for p in products]:
                    products.append({'name': text})
        
        data = {
            'url': url,
            'title': page_title,
            'description': description,
            'headings': headings[:30],  # Limit to first 30 headings
            'paragraphs': paragraphs[:20],  # Limit to first 20 paragraphs
            'links_count': len(links),
            'products': products[:15],  # Limit to first 15 products
            'status': 'success'
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the website: {e}")
        return {
            'url': url,
            'status': 'error',
            'error': str(e)
        }
    except Exception as e:
        print(f"Error processing the website: {e}")
        return {
            'url': url,
            'status': 'error',
            'error': str(e)
        }


def generate_summary(data: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary from scraped data.
    
    Args:
        data: Dictionary containing scraped data
        
    Returns:
        Summary string
    """
    if data.get('status') == 'error':
        return f"Error: Unable to scrape website. {data.get('error', 'Unknown error')}"
    
    summary = []
    summary.append("=" * 80)
    summary.append("WEBSITE SUMMARY: GE HEALTHCARE PATIENT MONITORING")
    summary.append("=" * 80)
    summary.append("")
    
    # Page Title
    summary.append(f"Page Title: {data['title']}")
    summary.append("")
    
    # Meta Description
    if data.get('description'):
        summary.append(f"Description: {data['description']}")
        summary.append("")
    
    # Main Headings
    if data.get('headings'):
        summary.append("Main Headings:")
        summary.append("-" * 40)
        h1_headings = [h for h in data['headings'] if h['level'] == 'h1']
        h2_headings = [h for h in data['headings'] if h['level'] == 'h2']
        
        if h1_headings:
            for h in h1_headings[:5]:
                summary.append(f"  • {h['text']}")
        
        if h2_headings:
            summary.append("\n  Key Sections:")
            for h in h2_headings[:10]:
                summary.append(f"    - {h['text']}")
        summary.append("")
    
    # Products
    if data.get('products'):
        summary.append("Products/Items Found:")
        summary.append("-" * 40)
        for product in data['products'][:10]:
            summary.append(f"  • {product['name']}")
        summary.append("")
    
    # Content Overview
    if data.get('paragraphs'):
        summary.append("Content Overview:")
        summary.append("-" * 40)
        # Take first few paragraphs for overview
        overview_text = ' '.join(data['paragraphs'][:3])
        if len(overview_text) > 500:
            overview_text = overview_text[:497] + "..."
        summary.append(overview_text)
        summary.append("")
    
    # Statistics
    summary.append("Statistics:")
    summary.append("-" * 40)
    summary.append(f"  • Total headings found: {len(data.get('headings', []))}")
    summary.append(f"  • Total paragraphs found: {len(data.get('paragraphs', []))}")
    summary.append(f"  • Total links found: {data.get('links_count', 0)}")
    summary.append(f"  • Products identified: {len(data.get('products', []))}")
    summary.append("")
    
    summary.append("=" * 80)
    
    return '\n'.join(summary)


def main():
    """Main function to run the scraper."""
    parser = argparse.ArgumentParser(
        description='Scrape and summarize GE Healthcare Patient Monitoring website',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s              # Scrape the actual website
  %(prog)s --demo       # Run with demo/mock data (for testing)
  %(prog)s --output result.json  # Save to custom file
        '''
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Use demo/mock data instead of scraping the actual website'
    )
    parser.add_argument(
        '--output',
        default='gehealthcare_scrape_result.json',
        help='Output JSON file (default: gehealthcare_scrape_result.json)'
    )
    
    args = parser.parse_args()
    
    url = "https://www.gehealthcare.in/products/patient-monitoring"
    
    print("GE Healthcare Patient Monitoring Website Scraper")
    print("=" * 80)
    
    if args.demo:
        print("Running in DEMO mode with mock data...")
        print()
        data = get_demo_data()
    else:
        print()
        # Scrape the website
        data = scrape_website(url)
    
    # Generate summary
    summary = generate_summary(data)
    print(summary)
    
    # Save to JSON file
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed data saved to: {args.output}")
    except Exception as e:
        print(f"\nWarning: Could not save data to file: {e}")
    
    return 0 if data.get('status') == 'success' else 1


if __name__ == "__main__":
    sys.exit(main())
