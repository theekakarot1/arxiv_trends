from langchain_community.document_loaders import ArxivLoader
import arxiv
import time
from datetime import datetime

arxiv_client = arxiv.Client()

def arxiv_document_loader(id_list):
    doc_list = []
    doc_details = []
    for id in id_list:
        try:
            loader = ArxivLoader(
                query=id,
                load_max_docs=1,
                load_all_available_meta=True,
                load_full_documents=True
            )
            doc = loader.load()
            doc_list.append(doc)
            temp_dict = {}
            if len(doc[0].page_content) > 1000000:
                continue
            for key in doc[0].metadata:
                temp_dict[key] = doc[0].metadata[key]
            temp_dict['page_content'] = doc[0].page_content
            doc_details.append(temp_dict)
        except Exception as e:
            print(e)
    return doc_list, doc_details

def safe_search(query, max_retries=3, sleep_time=5):
    """Run arxiv search with retries on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            search = arxiv.Search(
                query,
                max_results=100,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            return list(arxiv_client.results(search))
        except Exception as e:
            print(f"[Attempt {attempt}] Error: {e}")
            if attempt < max_retries:
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached, skipping this query.")
                return []


def get_arxiv_documents():
    id_list = []
    date_ranges = [
        # 2017
        ("20170101", "20170228"),
        ("20170301", "20170430"),
        ("20170501", "20170630"),
        ("20170701", "20170831"),
        ("20170901", "20171031"),
        ("20171101", "20171231"),

        # 2018
        ("20180101", "20180228"),
        ("20180301", "20180430"),
        ("20180501", "20180630"),
        ("20180701", "20180831"),
        ("20180901", "20181031"),
        ("20181101", "20181231"),

        # 2019
        ("20190101", "20190228"),
        ("20190301", "20190430"),
        ("20190501", "20190630"),
        ("20190701", "20190831"),
        ("20190901", "20191031"),
        ("20191101", "20191231"),

        # 2020
        ("20200101", "20200229"),
        ("20200301", "20200430"),
        ("20200501", "20200630"),
        ("20200701", "20200831"),
        ("20200901", "20201031"),
        ("20201101", "20201231"),

        # 2021
        ("20210101", "20210228"),
        ("20210301", "20210430"),
        ("20210501", "20210630"),
        ("20210701", "20210831"),
        ("20210901", "20211031"),
        ("20211101", "20211231"),

        # 2022
        ("20220101", "20220228"),
        ("20220301", "20220430"),
        ("20220501", "20220630"),
        ("20220701", "20220831"),
        ("20220901", "20221031"),
        ("20221101", "20221231"),

        # 2023
        ("20230101", "20230228"),
        ("20230301", "20230430"),
        ("20230501", "20230630"),
        ("20230701", "20230831"),
        ("20230901", "20231031"),
        ("20231101", "20231231"),

        # 2024
        ("20240101", "20240229"),
        ("20240301", "20240430"),
        ("20240501", "20240630"),
        ("20240701", "20240831"),
        ("20240901", "20241031"),
        ("20241101", "20241231"),

        # 2025
        ("20250101", "20250228"),
        ("20250301", "20250430"),
    ]
    try:
        for start_date, end_date in date_ranges:

            results_chunk = safe_search(f"cat:cs.AI AND submittedDate:[{start_date} TO {end_date}]" )
            
            if not results_chunk:
                # If the chunk is empty, we've reached the end of the results.
                print("Reached the end of the results. Stopping pagination.")
                # break
                
            # print(f"Fetched {len(results_chunk)} results from index {start_index}.")
            humanReadableStartDate= datetime.strptime(start_date, "%Y%m%d").strftime("%B %d, %Y")
            humanReadableEndDate = datetime.strptime(end_date, "%Y%m%d").strftime("%B %d, %Y")
            print(f"Fetched {len(results_chunk)} results from {humanReadableStartDate} TO {humanReadableEndDate}.")

            for result in results_chunk:
                pdf_link_exists = any(link.title == 'pdf' for link in result.links)
                if pdf_link_exists:
                    id_list.append(result.entry_id.split('abs/')[1])
        doc_list, doc_details = arxiv_document_loader(list(set(id_list)))
        return doc_list, doc_details

    except Exception as e:
        print(f"Error during pagination: {e}")