import os
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


# CONFIGURATION

BASE_URL_NO_PAGE = (
    "https://www.dlib.si/results/"
    "?query=%27keywords%3dilustrirani+slovenec%27"
    "&frelation=Ilustrirani+Slovenec"
    "&sortDir=ASC"
    "&sort=date"
    "&pageSize=100"
)

BASE_URL = BASE_URL_NO_PAGE + "&page={page}"

DOWNLOAD_DIR = "downloaded_pdfs"
IMAGES_DIR = "images"
MAX_RESULT_PAGES = None


# HELPERS

def extract_identifier(url):
    """
    Extracts a URN or ISBN identifier from a dLib URL.
    Example:
        URN:NBN:SI:DOC-ABC123
    """

    parsed_url = urlparse(url)
    path_segments = parsed_url.path.split("/")

    identifier = next(
        (
            segment
            for segment in path_segments
            if segment.startswith(("URN:", "ISBN:"))
        ),
        None
    )

    return identifier


# FIND + DOWNLOAD PDFs

def download_pdfs(result_page_url, download_dir, session):
    """
    Finds Ilustrirani Slovenec issue pages on a dLib result page,
    downloads their corresponding PDFs, and returns the local PDF paths.
    """

    print("\n============================================================")
    print(f"Fetching result page:")
    print(result_page_url)
    print("============================================================")

    os.makedirs(download_dir, exist_ok=True)

    try:
        response = session.get(
            result_page_url,
            timeout=30
        )
        response.raise_for_status()

    except requests.RequestException as e:
        print(f"[ERROR] Could not fetch result page: {e}")
        return []

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    # Find issue detail pages

    issue_links = set()

    for a_tag in soup.select('a[href*="/details/"]'):

        href = a_tag.get("href")

        if not href:
            continue

        issue_url = urljoin(
            result_page_url,
            href
        )

        issue_links.add(issue_url)

    issue_links = sorted(issue_links)

    print(
        f"Found {len(issue_links)} issue detail links."
    )

    if not issue_links:
        return []

    downloaded_files = []

    # Visit each issue page

    for index, issue_url in enumerate(
        issue_links,
        start=1
    ):

        print(
            f"\n[{index}/{len(issue_links)}] "
            f"Processing issue:"
        )
        print(issue_url)

        try:
            issue_response = session.get(
                issue_url,
                timeout=30
            )
            issue_response.raise_for_status()

        except requests.RequestException as e:
            print(
                f"[WARN] Could not open issue page: {e}"
            )
            continue

        issue_soup = BeautifulSoup(
            issue_response.content,
            "html.parser"
        )

        # Find PDF link

        pdf_link_tag = issue_soup.find(
            "a",
            href=True,
            string=lambda s:
                s is not None
                and "pdf" in s.lower()
        )

        # Fallback
        if pdf_link_tag is None:

            for a_tag in issue_soup.find_all(
                "a",
                href=True
            ):

                href = a_tag["href"]

                href_upper = href.upper()

                if (
                    href_upper.endswith("/PDF")
                    or "/PDF/" in href_upper
                ):
                    pdf_link_tag = a_tag
                    break

        if pdf_link_tag is None:
            print(
                "[WARN] No PDF link found."
            )
            continue

        pdf_url = urljoin(
            issue_url,
            pdf_link_tag["href"]
        )

        # Determine stable filename

        identifier = extract_identifier(pdf_url)

        if identifier is None:

            # fallback: try issue page URL
            identifier = extract_identifier(
                issue_url
            )

        if identifier is None:
            print(
                f"[WARN] Could not extract URN/ISBN "
                f"from {pdf_url}"
            )
            continue

        filename = (
            identifier.replace(":", "_")
            + ".pdf"
        )

        save_path = os.path.join(
            download_dir,
            filename
        )

        # Resume existing valid download

        if (
            os.path.exists(save_path)
            and os.path.getsize(save_path) > 1024
        ):

            print(
                f"Already downloaded: {filename}"
            )

            downloaded_files.append(
                save_path
            )

            continue

        # Download PDF

        print(
            f"Downloading: {filename}"
        )

        try:
            pdf_response = session.get(
                pdf_url,
                stream=True,
                timeout=120,
                allow_redirects=True
            )

            pdf_response.raise_for_status()

        except requests.RequestException as e:
            print(
                f"[WARN] PDF download failed: {e}"
            )
            continue

        content_type = (
            pdf_response.headers
            .get("Content-Type", "")
            .lower()
        )

        if "application/pdf" not in content_type:

            print(
                "[WARN] URL did not return a PDF."
            )

            print(
                f"Content-Type: {content_type}"
            )

            continue

        try:
            with open(
                save_path,
                "wb"
            ) as f:

                for chunk in pdf_response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        f.write(chunk)

        except Exception as e:

            print(
                f"[WARN] Failed while saving "
                f"{filename}: {e}"
            )

            if os.path.exists(save_path):
                os.remove(save_path)

            continue

        # Validate PDF header

        try:
            with open(
                save_path,
                "rb"
            ) as f:

                header = f.read(5)

        except Exception as e:

            print(
                f"[WARN] Could not validate "
                f"{filename}: {e}"
            )

            continue

        if header != b"%PDF-":

            print(
                f"[WARN] Invalid PDF downloaded. "
                f"Deleting {filename}"
            )

            os.remove(save_path)

            continue

        print(
            f"Saved: {save_path}"
        )

        downloaded_files.append(
            save_path
        )

    return downloaded_files


# PDF -> INDIVIDUAL PAGE IMAGES

def split_pdf_into_pages(
    pdf_path,
    output_dir
):
    """
    Converts every page of a PDF into an individual PNG image
    at 300 DPI.

    Example output:

    images/
        URN_NBN_SI_DOC-XXXX_page_001.png
        URN_NBN_SI_DOC-XXXX_page_002.png
        ...
    """

    pdf_filename = os.path.basename(
        pdf_path
    )

    base_name = os.path.splitext(
        pdf_filename
    )[0]

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    try:
        doc = fitz.open(
            pdf_path
        )

    except Exception as e:

        print(
            f"[ERROR] Could not open "
            f"{pdf_filename}: {e}"
        )

        return 0

    total_pages = len(doc)

    print(
        f"\nSplitting {pdf_filename}"
    )

    print(
        f"Pages: {total_pages}"
    )

    new_pages = 0

    try:

        for page_index in range(
            total_pages
        ):

            image_filename = (
                f"{base_name}"
                f"_page_"
                f"{page_index + 1:03d}"
                f".png"
            )

            image_path = os.path.join(
                output_dir,
                image_filename
            )

            # Resume existing valid PNG

            if (
                os.path.exists(image_path)
                and os.path.getsize(image_path) > 1000
            ):
                continue

            page = doc.load_page(
                page_index
            )

            pix = page.get_pixmap(
                dpi=300,
                alpha=False
            )

            pix.save(
                image_path
            )

            new_pages += 1

    except Exception as e:

        print(
            f"[ERROR] Failed while processing "
            f"{pdf_filename}: {e}"
        )

    finally:

        doc.close()

    print(
        f"Created {new_pages} new images."
    )

    return total_pages


# COUNT FILES

def count_files(
    folder,
    extension
):

    if not os.path.isdir(folder):
        return 0

    return sum(
        1
        for file in os.listdir(folder)
        if file.lower().endswith(
            extension.lower()
        )
    )


# MAIN

def main():

    print(
        "============================================================"
    )

    print(
        "Ilustrirani Slovenec Dataset Image Builder"
    )

    print(
        "============================================================"
    )

    os.makedirs(
        DOWNLOAD_DIR,
        exist_ok=True
    )

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120 Safari/537.36"
        )
    })

    all_pdfs = set()

    page_number = 1

    # STEP 1: DOWNLOAD ALL PDF ISSUES

    while True:

        if (
            MAX_RESULT_PAGES is not None
            and page_number > MAX_RESULT_PAGES
        ):
            break

        if page_number == 1:

            result_page_url = (
                BASE_URL_NO_PAGE
            )

        else:

            result_page_url = (
                BASE_URL.format(
                    page=page_number
                )
            )

        new_files = download_pdfs(
            result_page_url,
            DOWNLOAD_DIR,
            session
        )

        if not new_files:

            print(
                f"\nNo PDFs found on "
                f"result page {page_number}."
            )

            print(
                "Stopping pagination."
            )

            break

        before = len(all_pdfs)

        all_pdfs.update(
            new_files
        )

        after = len(all_pdfs)

        print(
            f"\nUnique PDFs collected so far: "
            f"{after}"
        )

        if (
            page_number > 1
            and after == before
        ):

            print(
                "No new PDFs found."
            )

            print(
                "Stopping pagination."
            )

            break

        page_number += 1

    # Also pick up PDFs already present locally

    for filename in os.listdir(
        DOWNLOAD_DIR
    ):

        if filename.lower().endswith(
            ".pdf"
        ):

            all_pdfs.add(
                os.path.join(
                    DOWNLOAD_DIR,
                    filename
                )
            )

    all_pdfs = sorted(
        all_pdfs
    )

    print(
        "\n============================================================"
    )

    print(
        f"PDFs available: "
        f"{len(all_pdfs)}"
    )

    print(
        "============================================================"
    )

    if not all_pdfs:

        print(
            "No PDFs found. Exiting."
        )

        return

    # STEP 2: SPLIT PDFs INTO PAGE IMAGES

    expected_pages = 0

    for index, pdf_path in enumerate(
        all_pdfs,
        start=1
    ):

        print(
            f"\n[{index}/{len(all_pdfs)}]"
        )

        num_pages = split_pdf_into_pages(
            pdf_path,
            IMAGES_DIR
        )

        expected_pages += num_pages

    # FINAL SUMMARY

    pdf_count = count_files(
        DOWNLOAD_DIR,
        ".pdf"
    )

    image_count = count_files(
        IMAGES_DIR,
        ".png"
    )

    print(
        "\n============================================================"
    )

    print(
        "DONE"
    )

    print(
        "============================================================"
    )

    print(
        f"Downloaded PDFs: {pdf_count}"
    )

    print(
        f"Expected pages from PDFs: "
        f"{expected_pages}"
    )

    print(
        f"PNG page images created: "
        f"{image_count}"
    )

    print(
        f"\nPDF folder:"
        f"\n  {os.path.abspath(DOWNLOAD_DIR)}"
    )

    print(
        f"\nImage folder:"
        f"\n  {os.path.abspath(IMAGES_DIR)}"
    )


if __name__ == "__main__":
    main()