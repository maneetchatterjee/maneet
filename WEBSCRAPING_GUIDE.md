# Web Scraping Documentation

## GE Healthcare Patient Monitoring Scraper

This script scrapes the GE Healthcare Patient Monitoring website and generates a comprehensive summary of its content.

### Location
`scripts/scrape_gehealthcare.py`

### Features

- **Web Scraping**: Extracts content from the GE Healthcare patient monitoring products page
- **Structured Data Extraction**: Captures titles, headings, paragraphs, links, and product information
- **Summary Generation**: Creates a human-readable summary of the website content
- **JSON Export**: Saves detailed scraped data in JSON format for further analysis
- **Demo Mode**: Includes mock data for testing without internet access

### Requirements

The following dependencies are required (already added to `requirements.txt`):
- `beautifulsoup4>=4.9.0`
- `requests>=2.25.0`

Install with:
```bash
pip install -r requirements.txt
```

### Usage

#### Basic Usage (Scrape Actual Website)
```bash
python scripts/scrape_gehealthcare.py
```

#### Demo Mode (Use Mock Data)
```bash
python scripts/scrape_gehealthcare.py --demo
```

#### Custom Output File
```bash
python scripts/scrape_gehealthcare.py --demo --output my_results.json
```

#### Help
```bash
python scripts/scrape_gehealthcare.py --help
```

### Output

The script produces two outputs:

1. **Console Summary**: A formatted summary displayed in the terminal with:
   - Page title and description
   - Main headings and sections
   - Products/items found
   - Content overview
   - Statistics (headings, paragraphs, links count)

2. **JSON File**: Detailed structured data saved to a JSON file containing:
   - URL
   - Page title
   - Meta description
   - All headings with hierarchy
   - Paragraph content
   - Link information
   - Product details
   - Status information

### Example Output

```
================================================================================
WEBSITE SUMMARY: GE HEALTHCARE PATIENT MONITORING
================================================================================

Page Title: Patient Monitoring | GE HealthCare India

Description: Explore GE HealthCare's range of patient monitoring solutions...

Main Headings:
----------------------------------------
  • Patient Monitoring Solutions

  Key Sections:
    - Comprehensive Patient Monitoring
    - Vital Signs Monitoring
    - Multi-Parameter Monitors
    ...

Products/Items Found:
----------------------------------------
  • CARESCAPE B850 Patient Monitor
  • CARESCAPE B650 Patient Monitor
  • Dash Series Patient Monitors
  ...

Statistics:
----------------------------------------
  • Total headings found: 10
  • Total paragraphs found: 5
  • Total links found: 45
  • Products identified: 10
```

### Website Summary

Based on the scraped data, the **GE Healthcare Patient Monitoring** website showcases:

**Main Purpose**: The website presents GE HealthCare's comprehensive portfolio of patient monitoring solutions designed to help healthcare professionals make informed clinical decisions.

**Key Product Categories**:
1. **Multi-parameter Patient Monitors** (CARESCAPE B850, B650, B450)
2. **Vital Signs Monitors**
3. **Anesthesia Monitoring Solutions**
4. **Telemetry Systems** (ApexPro Series)
5. **Central Monitoring Stations**
6. **Connectivity Solutions** (CARESCAPE Gateway)
7. **Neonatal Monitoring**

**Core Capabilities**:
- Real-time vital signs monitoring
- Advanced analytics and clinical decision support
- Hospital information system integration
- Alarm management and early warning scores
- Remote monitoring capabilities
- Workflow efficiency enhancement

**Target Environments**:
- Intensive Care Units (ICU)
- Operating Rooms
- Emergency Departments
- General Wards
- Neonatal Care Units

**Key Value Propositions**:
- Accurate and reliable clinical data
- Improved patient outcomes
- Enhanced care coordination
- Reduced alarm fatigue
- Scalable and flexible solutions
- Seamless data management

The website emphasizes GE HealthCare's commitment to providing comprehensive monitoring solutions that support healthcare professionals in delivering high-quality patient care across various clinical settings.

### Technical Details

**Scraping Strategy**:
- Uses BeautifulSoup4 for HTML parsing
- Extracts structured content (headings, paragraphs, links)
- Identifies product information through pattern matching
- Handles errors gracefully with informative messages

**Data Structure**:
```json
{
  "url": "string",
  "title": "string",
  "description": "string",
  "headings": [{"level": "string", "text": "string"}],
  "paragraphs": ["string"],
  "links_count": "number",
  "products": [{"name": "string"}],
  "status": "success|error"
}
```

### Troubleshooting

**Network Access Issues**:
If you encounter network connectivity issues, use the `--demo` flag to run with mock data:
```bash
python scripts/scrape_gehealthcare.py --demo
```

**Dependencies Not Installed**:
Make sure all dependencies are installed:
```bash
pip install beautifulsoup4 requests
```

### Notes

- The script includes a demo mode with representative data for testing purposes
- When scraping live websites, respect robots.txt and rate limits
- The script uses a user-agent header to mimic browser requests
- Timeout is set to 30 seconds for network requests
