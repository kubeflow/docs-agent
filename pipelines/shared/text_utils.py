import re

def clean_content(content: str) -> str:
    """Aggressively cleans text content for better embedding quality."""
    # Remove Hugo frontmatter (both --- and +++ styles)
    content = re.sub(r'^\s*[+\-]{3,}.*?[+\-]{3,}\s*', '', content, flags=re.DOTALL | re.MULTILINE)

    # Remove Hugo template syntax
    content = re.sub(r'\{\{.*?\}\}', '', content, flags=re.DOTALL)

    # Remove HTML comments and tags
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'<[^>]+>', ' ', content)

    # Remove navigation/menu artifacts
    content = re.sub(r'\b(Get Started|Contribute|GenAI|Home|Menu|Navigation)\b', '', content, flags=re.IGNORECASE)

    # Clean up URLs and links
    content = re.sub(r'https?://[^\s]+', '', content)
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # Convert [text](url) to text

    # Remove excessive whitespace and normalize
    content = re.sub(r'\s+', ' ', content)  # Multiple spaces to single
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines to double
    content = content.strip()
    
    return content
