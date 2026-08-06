import re

def clean_content(content: str) -> str:
    """
    Clean documentation text before embedding.
    Removes Hugo frontmatter, templates, HTML tags,
    markdown artifacts, and normalizes whitespace.
    """

    # Remove Hugo frontmatter (--- or +++)
    content = re.sub(
        r'^\s*[+\-]{3,}.*?[+\-]{3,}\s*',
        '',
        content,
        flags=re.DOTALL | re.MULTILINE
    )

    # Remove Hugo template syntax
    content = re.sub(r'\{\{.*?\}\}', '', content, flags=re.DOTALL)

    # Remove HTML comments and tags
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'<[^>]+>', ' ', content)

    # Remove navigation/menu artifacts
    content = re.sub(
        r'\b(Get Started|Contribute|GenAI|Home|Menu|Navigation)\b',
        '',
        content,
        flags=re.IGNORECASE
    )

    # Remove URLs
    content = re.sub(r'https?://[^\s]+', '', content)

    # Convert markdown links
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

    return content.strip()
