from kfp import dsl
from kfp.dsl import Output, Dataset

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["requests", "beautifulsoup4"]
)
def github_scraper_component(
    repo_owner: str,
    repo_name: str,
    directory_path: str,
    github_token: str,
    output_dataset: Output[Dataset]
):
    import requests
    import json
    import base64
    from bs4 import BeautifulSoup
    import os

    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory_path}"

    def get_files_recursive(url):
        files = []
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            items = response.json()

            for item in items:
                if item['type'] == 'file' and (item['name'].endswith('.md') or item['name'].endswith('.html')):
                    file_response = requests.get(item['url'], headers=headers)
                    file_response.raise_for_status()
                    file_data = file_response.json()
                    content = base64.b64decode(file_data['content']).decode('utf-8')

                    # Extract text from HTML files
                    if item['name'].endswith('.html'):
                        soup = BeautifulSoup(content, 'html.parser')
                        content = soup.get_text(separator=' ', strip=True)

                    files.append({
                        'path': item['path'],
                        'content': content,
                        'file_name': item['name'],
                        'repo': f"{repo_owner}/{repo_name}"
                    })
                elif item['type'] == 'dir':
                    files.extend(get_files_recursive(item['url']))
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return files

    files = get_files_recursive(api_url)
    print(f"Downloaded {len(files)} files")

    with open(output_dataset.path, 'w', encoding='utf-8') as f:
        for file_data in files:
            f.write(json.dumps(file_data, ensure_ascii=False) + '\n')
