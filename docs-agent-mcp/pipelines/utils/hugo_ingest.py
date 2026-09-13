import re
import toml
import yaml
from bs4 import BeautifulSoup

def parse_frontmatter(content):
    meta = {}
    body = content
    if content.startswith('+++'):
        try:
            end = content.index('+++', 3)
            meta = toml.loads(content[3:end])
            body = content[end+3:]
        except Exception:
            pass
    elif content.startswith('---'):
        try:
            end = content.index('---', 3)
            meta = yaml.safe_load(content[3:end]) or {}
            body = content[end+3:]
        except Exception:
            pass
    return meta, body

def process_html_table(html):
    soup = BeautifulSoup(html, 'html.parser')
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if not rows:
            continue
        
        # Determine max columns
        max_cols = 0
        for row in rows:
            cols = row.find_all(['th', 'td'])
            cols_count = sum(int(c.get('colspan', 1)) for c in cols)
            max_cols = max(max_cols, cols_count)
            
        grid = [['' for _ in range(max_cols)] for _ in range(len(rows))]
        
        for i, row in enumerate(rows):
            cols = row.find_all(['th', 'td'])
            col_idx = 0
            for col in cols:
                # Find next available cell
                while col_idx < max_cols and grid[i][col_idx] != '':
                    col_idx += 1
                if col_idx >= max_cols:
                    break
                
                rowspan = int(col.get('rowspan', 1))
                colspan = int(col.get('colspan', 1))
                text = col.get_text(separator=' ', strip=True)
                
                for r in range(rowspan):
                    for c in range(colspan):
                        if i + r < len(grid) and col_idx + c < max_cols:
                            grid[i + r][col_idx + c] = text
                col_idx += colspan
        
        # Reconstruct as markdown table
        md_table = []
        for row_data in grid:
            md_table.append("| " + " | ".join(row_data) + " |")
        
        # Replace the HTML table with the markdown table
        new_text = "\n" + "\n".join(md_table) + "\n"
        table.replace_with(soup.new_string(new_text))
        
    return str(soup)

def clean_hugo_markdown(content):
    meta, body = parse_frontmatter(content)
    
    stashes = {}
    
    # 2. STASH fenced blocks
    def stash_fence(m):
        k = f"%%FENCE{len(stashes)}%%"
        stashes[k] = m.group(0)
        return k
    body = re.sub(r'```.*?```', stash_fence, body, flags=re.DOTALL)
    
    # 3. STASH inline code
    def stash_code(m):
        k = f"%%CODE{len(stashes)}%%"
        stashes[k] = m.group(0)
        return k
    body = re.sub(r'`[^`\n]+`', stash_code, body)
    
    # 4. STASH GFM pipe tables
    def stash_gfm(m):
        k = f"%%GFM{len(stashes)}%%"
        stashes[k] = m.group(0)
        return k
    # simple table regex (multiple lines with pipes)
    body = re.sub(r'(?:\|.*\|[\r\n]+)+', stash_gfm, body)

    # Preserve placeholder tokens such as <YOUR_HF_TOKEN>; BeautifulSoup would
    # otherwise interpret these uppercase values as HTML tags.
    body = re.sub(r'<[A-Z][A-Z0-9_:-]*>', stash_code, body)
    
    # 5. Expand shortcodes (simple version)
    body = re.sub(r'\{\{%\s*alert.*?%\}\}(.*?)\{\{%\s*/alert\s*%\}\}', r'NOTE: \1', body, flags=re.DOTALL)
    body = re.sub(r'\{\{.*?\}\}', '', body, flags=re.DOTALL)
    
    # 6. Icons
    body = body.replace('fa-check', 'yes').replace('fa-xmark', 'no')
    
    # 7. Flatten HTML tables and HTML tags
    if '<table' in body:
        body = process_html_table(body)
    
    # 8. Images
    body = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', r'Figure: \1', body)
    
    # Remove remaining HTML tags
    soup = BeautifulSoup(body, 'html.parser')
    body = soup.get_text(separator=' ', strip=False)
    
    # 9. Markdown links
    body = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', body)
    
    # 10. Collapse horizontal whitespace
    body = re.sub(r'[ \t]+', ' ', body)
    
    # 11. Restore stashes
    for k, v in stashes.items():
        body = body.replace(k, v)
        
    return meta, body
