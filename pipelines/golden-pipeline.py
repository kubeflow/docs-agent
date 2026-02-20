import kfp
from kfp import dsl
from components.scraper import github_scraper_component
from components.processor import document_processor_component
from components.indexer import milvus_indexer_component

@dsl.pipeline(
    name="golden-data-ingestion",
    description="Established best-practice pipeline for indexing Kubeflow reference architecture and docs."
)
def golden_pipeline(
    github_token: str = "",
    milvus_host: str = "milvus-standalone.kubeflow.svc.cluster.local",
    milvus_port: str = "19530",
    collection_name: str = "golden_docs"
):
    # 1. Scrape Website Docs
    website_docs = github_scraper_component(
        repo_owner="kubeflow",
        repo_name="website",
        directory_path="content/en/docs",
        github_token=github_token
    )

    # 2. Scrape Manifests (Architecture Reference)
    manifest_docs = github_scraper_component(
        repo_owner="kubeflow",
        repo_name="manifests",
        directory_path="docs",
        github_token=github_token
    )

    # 3. Process Website Docs
    processed_website = document_processor_component(
        input_dataset=website_docs.outputs["output_dataset"],
        base_url="https://www.kubeflow.org/docs"
    )

    # 4. Process Manifest Docs
    processed_manifests = document_processor_component(
        input_dataset=manifest_docs.outputs["output_dataset"],
        base_url="https://github.com/kubeflow/manifests/blob/main/docs"
    )

    # 5. Index Website Docs
    milvus_indexer_component(
        processed_dataset=processed_website.outputs["output_dataset"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name
    )

    # 6. Index Manifest Docs
    milvus_indexer_component(
        processed_dataset=processed_manifests.outputs["output_dataset"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name
    )

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=golden_pipeline,
        package_path="golden_pipeline.yaml"
    )
