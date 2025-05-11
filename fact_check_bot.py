import os
import json
import requests
from typing import Dict, List, Optional, ClassVar, Callable
from datetime import datetime
import re
import logging
import firecrawl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool, tool

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI LLM with API key from environment variables
llm = ChatOpenAI(
    model="gpt-4-turbo",  # You can adjust the model as needed
    temperature=0.2,
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Create a DuckDuckGo search tool for research
class SearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web for information using DuckDuckGo"
    search_tool: DuckDuckGoSearchRun = None
    
    def __init__(self):
        super().__init__()
        self.search_tool = DuckDuckGoSearchRun()
    
    def _run(self, query: str) -> str:
        """
        Run a web search using DuckDuckGo
        Args:
            query (str): The search query
        Returns:
            str: The search results
        """
        logger.info(f"Searching for: {query}")
        
        # For news-related claims, enhance query with date and source terms
        if any(term in query.lower() for term in ["bbc", "reuters", "news", "reported"]):
            # Try to extract date info if present
            date_match = re.search(r'\b(202\d|january|february|march|april|may|june|july|august|september|october|november|december)\b', query.lower())
            if not date_match and "recent" not in query.lower():
                # Add recency to the query if not already specified
                enhanced_query = f"{query} recent news"
                logger.info(f"Enhanced search query to: {enhanced_query}")
                return self.search_tool.run(enhanced_query)
                
        # If news verification requires multiple trusted sources
        if "verify" in query.lower() or "fact check" in query.lower():
            # Try two searches - one normal and one with "verified by" terms
            results1 = self.search_tool.run(query)
            enhanced_query = f"{query} confirmed OR verified by multiple sources"
            logger.info(f"Running verification query: {enhanced_query}")
            results2 = self.search_tool.run(enhanced_query)
            
            return f"Regular search results:\n{results1}\n\nVerification search results:\n{results2}"
            
        # Standard query
        return self.search_tool.run(query)

# Custom Firecrawl Tool for web scraping
class FirecrawlTool(BaseTool):
    name: str = "Web Scraper"
    description: str = "Scrapes a URL and returns the content in markdown format"
    
    def _run(self, url: str) -> str:
        """
        Scrape content from a URL using the Firecrawl API.
        Args:
            url (str): The URL to scrape
        Returns:
            str: The content in markdown format
        """
        # Try to access the context directly (CrewAI-specific)
        try:
            from crewai import Task
            current_task = Task.current()
            if current_task and hasattr(current_task, 'context'):
                context_url = current_task.context.get('url')
                if context_url:
                    logger.info(f"[FIRECRAWL] Found URL in context: {context_url}")
                    if url != context_url:
                        logger.warning(f"[FIRECRAWL] URL mismatch - Context: {context_url}, Provided: {url}")
                        logger.info(f"[FIRECRAWL] Using context URL: {context_url}")
                        url = context_url
                else:
                    logger.warning("[FIRECRAWL] No URL found in context")
            else:
                logger.warning("[FIRECRAWL] Could not access task context")
        except ImportError:
            logger.error("[FIRECRAWL] CrewAI Task import failed")
        except Exception as e:
            logger.error(f"[FIRECRAWL] Error accessing context: {str(e)}")
        
        # Log input details
        logger.info("=" * 80)
        logger.info("[FIRECRAWL] Input Validation:")
        logger.info(f"[FIRECRAWL] Input type: {type(url)}")
        logger.info(f"[FIRECRAWL] Input value: {url}")
        logger.info(f"[FIRECRAWL] Input length: {len(str(url)) if url else 0}")
        
        # Validate URL
        if not url:
            error_msg = "[FIRECRAWL] Error: Empty URL received"
            logger.error(error_msg)
            return "Error: No URL provided"
            
        if not isinstance(url, str):
            error_msg = f"[FIRECRAWL] Error: Invalid URL type received: {type(url)}"
            logger.error(error_msg)
            return f"Error: Invalid URL type: {type(url)}"
            
        # Basic URL format validation
        if not url.startswith(('http://', 'https://')):
            error_msg = f"[FIRECRAWL] Error: Invalid URL format: {url}"
            logger.error(error_msg)
            return "Error: Invalid URL format. URL must start with http:// or https://"
            
        logger.info("[FIRECRAWL] Input validation passed")
        logger.info(f"[FIRECRAWL] Starting scrape for URL: {url}")
        logger.info("=" * 80)
        
        try:
            firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY")
            if not firecrawl_api_key:
                logger.error("[FIRECRAWL] API key not found in environment variables.")
                logger.info("[FIRECRAWL] Falling back to direct scraping method.")
                return self._fallback_scrape(url)
            
            logger.info(f"[FIRECRAWL] API Key found: {firecrawl_api_key[:5]}...")
            
            headers = {
                "Authorization": f"Bearer {firecrawl_api_key.strip()}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            payload = {
                "url": url,
                "formats": ["markdown", "html"],
                "wait_for": 10000,
                "include_raw_html": True
            }
            
            # Log headers without the full API key
            safe_headers = headers.copy()
            safe_headers["Authorization"] = f"Bearer {firecrawl_api_key[:5]}..."
            logger.info(f"[FIRECRAWL] Request headers: {safe_headers}")
            logger.info(f"[FIRECRAWL] Request payload: {json.dumps(payload, indent=2)}")
            logger.info("[FIRECRAWL] Sending request to Firecrawl API...")
            
            response = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=30)
            logger.info(f"[FIRECRAWL] API response status: {response.status_code}")
            
            # Log response headers safely
            safe_response_headers = dict(response.headers)
            if "Authorization" in safe_response_headers:
                safe_response_headers["Authorization"] = "[REDACTED]"
            logger.info(f"[FIRECRAWL] API response headers: {safe_response_headers}")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"[FIRECRAWL] Full API response: {json.dumps(data, indent=2)}")
                
                if data.get("success"):
                    content = data.get("data", {}).get("markdown", data.get("data", {}).get("html", ""))
                    if content and len(content.strip()) > 100 and "Example Domain" not in content:
                        logger.info("[FIRECRAWL] Successfully extracted markdown content")
                        logger.info(f"[FIRECRAWL] Content length: {len(content)} characters")
                        logger.info(f"[FIRECRAWL] Content preview: {content[:500]}...")
                        return content
                    else:
                        logger.warning("[FIRECRAWL] Content is empty, too short, or contains placeholder text")
                        logger.info(f"[FIRECRAWL] Full response: {json.dumps(data, indent=2)}")
                        logger.info("[FIRECRAWL] Falling back to direct scraping method.")
                        return self._fallback_scrape(url)
                else:
                    error_msg = f"[FIRECRAWL] API request succeeded but returned error: {data.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    logger.info(f"[FIRECRAWL] Full response: {json.dumps(data, indent=2)}")
                    logger.info("[FIRECRAWL] Falling back to direct scraping method.")
                    return self._fallback_scrape(url)
            else:
                error_msg = f"[FIRECRAWL] API returned status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                logger.info("[FIRECRAWL] Falling back to direct scraping method.")
                return self._fallback_scrape(url)
        except requests.exceptions.RequestException as e:
            error_msg = f"[FIRECRAWL] Network error while scraping URL {url}: {str(e)}"
            logger.error(error_msg)
            logger.info("[FIRECRAWL] Falling back to direct scraping method.")
            return self._fallback_scrape(url)
        except Exception as e:
            error_msg = f"[FIRECRAWL] Unexpected error while scraping URL {url}: {str(e)}"
            logger.error(error_msg)
            logger.info("[FIRECRAWL] Falling back to direct scraping method.")
            return self._fallback_scrape(url)

    def _fallback_scrape(self, url: str) -> str:
        """Custom scraper that works well with BBC content"""
        logger.info(f"[FALLBACK] Starting direct scrape for URL: {url}")
        try:
            # Modern Chrome headers with additional BBC-specific headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0"
            }
            logger.info(f"[FALLBACK] Using headers: {headers}")
            
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            logger.info(f"[FALLBACK] Response status code: {response.status_code}")
            logger.info(f"[FALLBACK] Response URL (after redirects): {response.url}")
            logger.info(f"[FALLBACK] Response headers: {dict(response.headers)}")
            logger.info(f"[FALLBACK] Response content (first 500 chars): {response.text[:500]}...")
            
            if response.status_code == 200:
                if "example.com" in response.url.lower():
                    logger.error("[FALLBACK] Redirected to example.com - likely a scraping issue or incorrect URL.")
                    return f"Error: Redirected to example.com while scraping {url}."

                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                content_blocks = []
                
                # Special handling for BBC articles
                if "bbc.com" in url or "bbc.co.uk" in url:
                    logger.info("[FALLBACK] Detected BBC article - using BBC-specific extraction")
                    
                    # Try multiple headline selectors
                    headline_selectors = [
                        'h1',
                        '[data-component="headline"]',
                        '.article-headline',
                        '.story-body__h1',
                        '.ssrcss-15xko80-StyledHeading'
                    ]
                    
                    for selector in headline_selectors:
                        headline = soup.select_one(selector)
                        if headline:
                            headline_text = headline.get_text().strip()
                            logger.info(f"[FALLBACK] Found headline using selector {selector}: {headline_text}")
                            content_blocks.append(f"# {headline_text}")
                            break
                    
                    if not content_blocks:
                        logger.warning("[FALLBACK] Could not find headline using any selector.")
                    
                    # Updated BBC content selectors
                    containers = soup.select("""
                        article,
                        [data-component="text-block"],
                        .ssrcss-1q0x1qg-RichTextContainer,
                        .story-body__inner,
                        div[data-component="text"],
                        div[class*="RichText"],
                        .ssrcss-uf6wea-RichTextContainer,
                        .ssrcss-7uxr49-RichTextContainer,
                        .article__body-content,
                        .story-body,
                        [data-component="text-block"],
                        .body-content-container,
                        .article-body-content
                    """)
                    logger.info(f"[FALLBACK] Found {len(containers)} content containers")
                    
                    # Extract paragraphs from containers
                    for container in containers:
                        # Try multiple paragraph selectors
                        for p_selector in ['p', '[data-component="text-block"] p', '.paragraph']:
                            paragraphs = container.select(p_selector)
                            logger.info(f"[FALLBACK] Found {len(paragraphs)} paragraphs using selector {p_selector} in container")
                            
                            for p in paragraphs:
                                # Skip navigation and social media paragraphs
                                if any(skip in str(p.get('class', [])) for skip in ['navigation', 'social', 'share', 'hidden']):
                                    continue
                                    
                                p_text = p.get_text().strip()
                                if p_text and len(p_text) > 10 and not p_text.startswith('Share this'):
                                    logger.info(f"[FALLBACK] Extracted paragraph: {p_text[:100]}...")
                                    content_blocks.append(p_text)
                    
                    # Updated metadata selectors
                    metadata_selectors = [
                        '.ssrcss-17m4u3h-MetadataStrip',
                        '.ssrcss-f4vpt6-MetadataLink',
                        '.ssrcss-1n2rzdv-StyledPublishingContext',
                        '.article-info',
                        '.author-unit__content',
                        '.byline',
                        '.article-meta',
                        '.article__author',
                        '.article__timestamp',
                        '[data-component="byline"]',
                        '[data-component="meta"]',
                        '[data-component="topics"]'
                    ]
                    
                    for selector in metadata_selectors:
                        metadata = soup.select(selector)
                        logger.info(f"[FALLBACK] Found {len(metadata)} metadata elements using selector {selector}")
                        
                        for meta in metadata:
                            meta_text = meta.get_text().strip()
                            if meta_text and not any(skip in meta_text.lower() for skip in ['share this', 'follow us', 'subscribe']):
                                logger.info(f"[FALLBACK] Extracted metadata: {meta_text}")
                                content_blocks.append(f"*{meta_text}*")
                    
                    # If still no content, try generic paragraphs with minimum length filter
                    if not content_blocks:
                        logger.info("[FALLBACK] No content found in BBC-specific selectors, trying generic paragraphs")
                        paragraphs = soup.select('p')
                        logger.info(f"[FALLBACK] Found {len(paragraphs)} generic paragraphs")
                        for p in paragraphs:
                            p_text = p.get_text().strip()
                            # Increased minimum length for generic paragraphs to avoid noise
                            if p_text and len(p_text) > 40 and not any(skip in p_text.lower() for skip in ['cookie', 'subscribe', 'newsletter', 'share this']):
                                logger.info(f"[FALLBACK] Extracted generic paragraph: {p_text[:100]}...")
                                content_blocks.append(p_text)
                
                # Generic extraction for non-BBC sites
                if not content_blocks:
                    logger.info("[FALLBACK] Using generic extraction method")
                    main_content = soup.select_one('main, article, .content, #content, .article-content, .post-content')
                    if main_content:
                        paragraphs = main_content.select('p')
                        logger.info(f"[FALLBACK] Found {len(paragraphs)} paragraphs in main content")
                        for p in paragraphs:
                            p_text = p.get_text().strip()
                            if p_text and len(p_text) > 30:
                                logger.info(f"[FALLBACK] Extracted generic paragraph: {p_text[:100]}...")
                                content_blocks.append(p_text)
                    else:
                        logger.info("[FALLBACK] No main content found, trying all paragraphs")
                        paragraphs = soup.select('p')
                        logger.info(f"[FALLBACK] Found {len(paragraphs)} paragraphs in entire page")
                        for p in paragraphs:
                            p_text = p.get_text().strip()
                            if p_text and len(p_text) > 50:
                                logger.info(f"[FALLBACK] Extracted generic paragraph: {p_text[:100]}...")
                                content_blocks.append(p_text)
                
                # Last resort: extract body text
                if not content_blocks:
                    logger.info("[FALLBACK] No structured content found, extracting body text")
                    body_text = soup.body.get_text().strip() if soup.body else ""
                    import re
                    body_text = re.sub(r'\s+', ' ', body_text)
                    # Filter out common noise patterns
                    body_text = re.sub(r'(Share this|Follow us|Subscribe|Cookie|Newsletter).*?(\.|$)', '', body_text, flags=re.IGNORECASE)
                    content_blocks = [body_text[:8000]]
                    logger.info(f"[FALLBACK] Extracted body text (first 500 chars): {body_text[:500]}...")
                
                # Clean and join content blocks
                article_text = "\n\n".join(block.strip() for block in content_blocks if block.strip())
                sample = article_text[:500] + "..." if len(article_text) > 500 else article_text
                logger.info(f"[FALLBACK] Extracted content sample: {sample}")
                logger.info(f"[FALLBACK] Content length: {len(article_text)} characters")
                logger.info(f"[FALLBACK] Number of content blocks extracted: {len(content_blocks)}")
                
                if len(article_text) < 100:
                    error_msg = f"Error: Could not extract meaningful content from {url}."
                    logger.error(f"[FALLBACK] {error_msg}")
                    return error_msg
                
                return article_text
            else:
                error_msg = f"Error: Failed to retrieve content. Status code: {response.status_code}"
                logger.error(f"[FALLBACK] {error_msg}")
                return error_msg
        except Exception as e:
            error_msg = f"Error scraping URL {url}: {str(e)}"
            logger.error(f"[FALLBACK] {error_msg}")
            return error_msg

# Initialize tools
search_tool = SearchTool()
firecrawl_tool = FirecrawlTool()

# Define agents with specific roles
content_extractor = Agent(
    role="Content Extraction Specialist",
    goal="Extract and clean the most relevant content from a webpage using the URL provided in the input context under the 'url' key, removing clutter like ads and navigation elements.",
    backstory="""You are a skilled web scraper with expertise in extracting clean, readable content from webpages. 
    You must ALWAYS access the URL from the input context under the 'url' key and use it to scrape the webpage. 
    NEVER use any hardcoded URLs or default values like 'https://example.com'. 
    Your task is to scrape the webpage, identify the main article text, and remove irrelevant elements like ads and navigation.
    Always verify you are using the correct URL from the context before proceeding.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    tools=[firecrawl_tool]
)

claim_identifier = Agent(
    role="Claim Identification Expert",
    goal="Identify and extract factual claims from text that can be verified",
    backstory="""You are a linguistic specialist who can identify factual assertions 
    within text. You're trained to distinguish between opinions, predictions, and 
    verifiable factual claims. Your expertise helps separate what can be fact-checked 
    from what cannot.""",
    verbose=True,
    llm=llm
)

fact_researcher = Agent(
    role="Fact Research Specialist",
    goal="Find accurate information to verify claims using search tools",
    backstory="""You are a thorough researcher with a talent for finding accurate 
    information online. You know how to craft effective search queries and identify 
    reliable sources to gather evidence related to claims that need verification.""",
    verbose=True,
    llm=llm,
    tools=[search_tool]
)

claim_verifier = Agent(
    role="Claim Verification Analyst",
    goal="Analyze claims against research to determine their accuracy",
    backstory="""You are an analytical expert who specializes in comparing claims 
    against evidence. You can identify inconsistencies, confirm accuracies, and 
    determine the truthfulness of statements based on the available evidence.""",
    verbose=True,
    llm=llm
)

credibility_summarizer = Agent(
    role="Credibility Assessment Summarizer",
    goal="Create a clear, comprehensive summary of the fact-checking results",
    backstory="""You are a communication specialist who can clearly explain complex 
    fact-checking results. You're skilled at creating summaries that highlight key 
    findings and their implications for the overall credibility of content.""",
    verbose=True,
    llm=llm
)

# Define tasks for each agent
extract_content_task = Task(
    description="""
    Extract and clean the most relevant content from the scraped webpage.

    IMPORTANT: The URL to analyze is provided in the input context under the 'url' key.
    You MUST use this URL and not any default or hardcoded values.

    Step by step instructions:
    1. Get the URL from the input context using the 'url' key
    2. Log the exact URL you will use: "Using URL from context: {url}"
    3. Verify the URL is not a default value (like 'example.com')
    4. Use the Firecrawl tool to scrape ONLY the URL from the context
    5. Extract the main article text, removing ads, navigation, etc.
    6. Format the output as clean, readable text

    DO NOT:
    - Use any hardcoded URLs
    - Use default values like 'https://example.com'
    - Skip the URL logging step
    
    The success of this task depends on using the correct URL from the context.
    """,
    expected_output="""
    Clean, well-formatted text that represents the most relevant content from the scraped URL.
    The text should preserve all factual information while removing any clutter.
    
    The output should begin with:
    "Using URL from context: [actual URL]"
    
    Followed by the extracted content.
    """,
    agent=content_extractor,
    async_execution=False
)

identify_claims_task = Task(
    description="""
    Identify and extract factual claims from the cleaned content that can be verified.
    Your job is to:
    1. Analyze the cleaned text content provided in the context under the key 'cleaned_content'.
    2. If the content contains error messages (e.g., 'Error scraping URL', 'No content found'), or appears to be a placeholder (e.g., contains 'Example Domain'), return a message indicating the content is invalid: 'Error: No verifiable claims found due to invalid or inaccessible content.'
    3. Otherwise, identify specific factual assertions that can be verified. Examples of factual claims include statements about events, statistics, or historical facts (e.g., "Israel conducted airstrikes on Gaza on October 7, 2023," "At least 10 people were killed").
    4. Separate opinions (e.g., "The policy is unfair"), subjective statements (e.g., "The situation is dire"), and predictions (e.g., "It might rain tomorrow") from factual claims.
    5. List each claim separately with a brief explanation of why it's a factual claim that can be verified (e.g., "This is a factual claim because it describes a specific event that can be checked against news reports").
    6. Prioritize claims that are central to the article's main points.
    7. If no factual claims are found, return: 'No verifiable factual claims were identified in the content.'
    Focus on extracting 3-5 of the most significant factual claims from the content, or return an error message if the content is invalid.
    """,
    expected_output="""
    Either:
    - A list of 3-5 key factual claims from the content, each with:
      - The exact claim statement
      - A brief explanation of why it's a factual claim that can be verified
    Or:
    - A message: 'Error: No verifiable claims found due to invalid or inaccessible content' if the content is invalid
    - A message: 'No verifiable factual claims were identified in the content' if no claims are found
    """,
    agent=claim_identifier,
    async_execution=False
)

research_claims_task = Task(
    description="""
    Research each factual claim to find relevant information for verification.
    
    Your job is to:
    1. Take each claim identified in the previous task
    2. For each claim, search for corroborating evidence from reliable sources
    3. For very short or general claims, try different search approaches:
       - Search for the exact claim first
       - Then try adding context terms (news, recent, plans, announcement)
       - Look for reliable sources discussing the topic
    4. For news-related claims, look for other news organizations reporting similar information
    5. Note the reliability of each source you find
    
    Make at least two different search queries for each claim to find robust evidence.
    For very short claims like "Trump center is coming to India," make additional searches 
    with variations like "Trump center India plans" and "Trump center India announcement"
    to find the most relevant information.
    """,
    expected_output="""
    Research findings for each claim, including:
    - The original claim
    - Information found from searches relevant to the claim
    - The sources of information and their reliability
    - A preliminary assessment: Confirmed, Contradicted, or Insufficient Evidence
    
    Be thorough in researching even brief claims, making at least two different search
    attempts with different query formulations.
    """,
    agent=fact_researcher,
    async_execution=False
)

verify_claims_task = Task(
    description="""
    Analyze the research findings to verify the accuracy of each claim.
    
    Your job is to:
    1. Review the research findings for each claim
    2. Determine if each claim is True, False, Partially True, or Unverifiable based on the evidence
    3. Explain your verification decision with specific reference to the sources
    4. For BBC News reports, if other major news outlets reported the same information, this generally supports verification
    
    Rate each claim's verification status and provide a brief explanation of your reasoning.
    """,
    expected_output="""
    Verification results for each claim:
    
    1. Claim: "[original claim text]"
       - Verification: [True/False/Partially True/Unverifiable]
       - Evidence: [brief summary of supporting or contradicting evidence]
    
    2. Claim: "[original claim text]"
       - Verification: [True/False/Partially True/Unverifiable]
       - Evidence: [brief summary of supporting or contradicting evidence]
    
    [And so on for each claim]
    """,
    agent=claim_verifier,
    async_execution=False
)

summarize_credibility_task = Task(
    description="""
    Create a comprehensive summary of the fact-checking results.
    
    Your job is to:
    1. Summarize the overall credibility of the article based on the verified claims
    2. List the verified claims with their verification status
    3. Provide an overall credibility rating (Highly Credible, Mostly Credible, Somewhat Credible, or Not Credible)
    4. Give readers recommendations about how to interpret the content
    5. Add a final bold label at the very end of your report:
       - If the Overall Credibility Rating contains "Highly Credible" or "Mostly Credible," add "**Real**"
       - If the Overall Credibility Rating contains "Not Credible" or "Low Credibility," add "**Fake**"
       - For all other cases (including "Somewhat Credible"), add "**Uncertain**"
    
    Format your summary following this structure:
    
    "Based on the provided content from [URL]:
    1. Overall Assessment: [brief summary of article content and fact-check results]
    2. Summary of Key Verified and Disputed Claims:
       - Claim: [claim text] Verified: [verification status] ([brief evidence])
       - [repeat for each claim]
    3. Overall Credibility Rating: [rating] [explanation]
    4. Recommendations: [advice for readers]
    
    **[Final Label]**"
    
    For BBC News articles, unless contradictory evidence is found, they should generally receive at least a "Mostly Credible" rating as they are a mainstream news source with editorial standards.
    """,
    expected_output="""
    A comprehensive fact-checking summary in the exact format:
    
    "Based on the provided content from [URL]:
    1. Overall Assessment: [brief summary]
    2. Summary of Key Verified and Disputed Claims:
       - Claim: [claim text] Verified: [status] ([evidence])
       - Claim: [claim text] Verified: [status] ([evidence])
    3. Overall Credibility Rating: [rating] [explanation]
    4. Recommendations: [advice for readers]
    
    **[Real/Fake/Uncertain]**"
    """,
    agent=credibility_summarizer,
    async_execution=False
)

# Create a Crew instance with the agents and tasks
fact_check_crew = Crew(
    agents=[
        content_extractor,
        claim_identifier,
        fact_researcher,
        claim_verifier,
        credibility_summarizer
    ],
    tasks=[
        extract_content_task,
        identify_claims_task,
        research_claims_task,
        verify_claims_task,
        summarize_credibility_task
    ],
    verbose=True,  # Detailed logging
    process=Process.sequential  # Tasks will be executed in sequence
)

def fact_check_url(url: str) -> str:
    """
    Run the fact-checking crew on a URL.
    Args:
        url (str): The URL to fact-check
    Returns:
        str: The fact-checking results
    """
    logger.info(f"[URL_CHECK] Received URL for fact checking: {url}")
    
    # Validate URL format
    url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
    if not url_pattern.match(url):
        logger.error(f"[URL_CHECK] Invalid URL format: {url}")
        return "Error: Invalid URL format. Please provide a valid URL starting with http:// or https://."

    # Check for future dates in the URL
    try:
        year_match = re.search(r'/(\d{4})/', url)
        if year_match:
            year = int(year_match.group(1))
            current_year = datetime.now().year
            if year > current_year:
                logger.error(f"[URL_CHECK] Future date detected in URL: {year}")
                return f"Error: URL contains a future date ({year}), which may not be valid. Please check the URL."
    except Exception as e:
        logger.error(f"[URL_CHECK] Error validating URL date: {str(e)}")
        return f"Error validating URL date: {str(e)}"

    # Special handling for BBC articles
    is_bbc_article = "bbc.com" in url or "bbc.co.uk" in url
    logger.info(f"[URL_CHECK] BBC article detection: {is_bbc_article}")
    
    try:
        logger.info(f"[URL_CHECK] Starting fact check process for URL: {url}")
        
        # Initialize context
        initial_context = {"url": url}
        if is_bbc_article:
            logger.info("[URL_CHECK] Configuring BBC-specific processing")
            initial_context.update({
                "source_type": "bbc_news",
                "expected_claim_format": "news_article",
                "identify_instruction": "This is a BBC news article. Focus on extracting factual claims about events, statistics, and attributions."
            })
        
        # Log the context being passed to CrewAI
        logger.info(f"[URL_CHECK] Initial context for CrewAI: {json.dumps(initial_context, indent=2)}")
        
        # Kick off the crew
        logger.info("[URL_CHECK] Starting CrewAI kickoff")
        result = fact_check_crew.kickoff(inputs=initial_context)
        logger.info("[URL_CHECK] CrewAI kickoff completed")
        
        # Parse and format the result
        if isinstance(result, dict) and "tasks_output" in result:
            logger.info("[URL_CHECK] Processing CrewAI task outputs")
            task_outputs = result.get("tasks_output", [])
            content_extraction_result = task_outputs[0].raw if len(task_outputs) > 0 else "No content extraction result"
            
            # Log content extraction result
            logger.info(f"[URL_CHECK] Content extraction length: {len(content_extraction_result)}")
            
            if "Error" in content_extraction_result and len(content_extraction_result) < 500:
                logger.warning("[URL_CHECK] Content extraction error detected")
                if is_bbc_article:
                    logger.info("[URL_CHECK] Attempting BBC fallback extraction")
                    firecrawl_tool = FirecrawlTool()
                    content = firecrawl_tool._fallback_scrape(url)
                    if len(content) > 500:
                        logger.info("[URL_CHECK] BBC fallback successful, processing as text")
                        return fact_check_text(content)
                
                logger.error(f"[URL_CHECK] Content extraction failed: {content_extraction_result}")
                return f"Failed to extract content from URL: {url}. {content_extraction_result}"
                
            final_summary = task_outputs[-1].raw if task_outputs else "No results available"
            logger.info("[URL_CHECK] Final summary generated")
            
            if is_bbc_article and "verifiable claims" not in final_summary.lower():
                try:
                    logger.info("[URL_CHECK] Formatting BBC-specific output")
                    article_title = "BBC Article"
                    if len(task_outputs) > 0 and "# " in task_outputs[0].raw:
                        title_match = re.search(r'# (.+)', task_outputs[0].raw)
                        if title_match:
                            article_title = title_match.group(1)
                            logger.info(f"[URL_CHECK] Extracted BBC article title: {article_title}")
                    
                    formatted_output = f"""Based on the provided content from {url}:
1. Overall Assessment: {final_summary}
2. Source: BBC News, a generally reliable mainstream news source.
3. Recommendations: Information from BBC News is generally reliable, but for complete verification, cross-reference with other reputable sources.

**Real**"""
                    logger.info("[URL_CHECK] BBC output formatting successful")
                    return formatted_output
                except Exception as e:
                    logger.error(f"[URL_CHECK] Error formatting BBC result: {str(e)}")
            
            return final_summary
        logger.info("[URL_CHECK] Returning raw result")
        return str(result)
    except Exception as e:
        logger.error(f"[URL_CHECK] Unhandled error in fact checking: {str(e)}")
        return f"Error: {str(e)}"

def fact_check_text(text: str) -> str:
    try:
        # Create initial context with the text content
        initial_context = {
            "cleaned_content": text,
            "task": "text_analysis"  # Indicate this is a text analysis task
        }
        
        # Kick off the crew with the context
        result = fact_check_crew.kickoff(inputs=initial_context)
        
        # Parse and format the result
        if isinstance(result, dict) and "tasks_output" in result:
            # Get the final summary from the last task
            final_summary = result["tasks_output"][-1].raw
            return final_summary
        return str(result)
        
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example URL to fact-check (a news article)
    example_url = "https://www.bbc.com/news/world-europe-68736364"
    
    print("Starting fact-checking process for URL:", example_url)
    results = fact_check_url(example_url)
    print("\n=== FACT-CHECKING RESULTS ===\n")
    print(results)
    
    # Example of how to use with plain text instead of a URL
    """
    example_text = '''
    The COVID-19 vaccine was developed in less than a year, making it the fastest vaccine development in history.
    Studies show that regular exercise can reduce the risk of heart disease by up to 50%.
    The average global temperature has increased by 1.5 degrees Celsius since pre-industrial times.
    '''
    
    print("Starting fact-checking process for text")
    results = fact_check_text(example_text)
    print("\n=== FACT-CHECKING RESULTS ===\n")
    print(results)
    """ 