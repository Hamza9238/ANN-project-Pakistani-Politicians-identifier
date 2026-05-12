import os
import io
import time
import base64
import requests

from PIL import Image
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


def download_google_images(
    queries,
    scrolls=30,
    scroll_pause=2
):
    """
    Download Google Images for multiple queries.

    Parameters:
    -----------
    queries : list[str]
        Search queries

    scrolls : int
        Number of page scrolls

    scroll_pause : int
        Delay after each scroll
    """

    # ==========================
    # CHROMIUM + CHROMEDRIVER
    # ==========================
    options = Options()

    # Uncomment if needed
    # options.add_argument("--headless=new")

    options.add_argument(
        "--start-maximized"
    )

    # Let Selenium Manager handle the driver automatically on Windows
    driver = webdriver.Chrome(
        options=options
    )

    # ==========================
    # PROCESS EACH QUERY
    # ==========================
    for query in queries:

        print("\n" + "=" * 50)
        print(f"Processing: {query}")
        print("=" * 50)

        query_url = quote(query)

        # Folder-safe name
        safe_query = (
            query.strip()
            .replace(" ", "_")
        )

        # ./raw_dataset/query_name
        folder_name = os.path.join(
            ".",
            "raw_dataset",
            safe_query
        )

        os.makedirs(
            folder_name,
            exist_ok=True
        )

        print(
            f"Saving images to: "
            f"{folder_name}"
        )

        # ==========================
        # OPEN GOOGLE IMAGES
        # ==========================
        url = (
            "https://www.google.com/"
            f"search?q={query_url}"
            "&tbm=isch"
        )

        driver.get(url)

        time.sleep(5)

        # ==========================
        # SCROLL
        # ==========================
        for _ in range(scrolls):

            driver.execute_script(
                "window.scrollTo("
                "0, "
                "document.body.scrollHeight"
                ");"
            )

            time.sleep(scroll_pause)

        print("Finished scrolling")

        # ==========================
        # FIND IMAGES
        # ==========================
        img_elements = (
            driver.find_elements(
                By.TAG_NAME,
                "img"
            )
        )

        print(
            f"Found "
            f"{len(img_elements)} "
            f"image elements"
        )

        # ==========================
        # DOWNLOAD
        # ==========================
        count = 0
        seen_urls = set()

        for img in img_elements:

            try:
                img_url = img.get_attribute(
                    "src"
                )

                if not img_url:
                    continue

                if img_url in seen_urls:
                    continue

                seen_urls.add(img_url)

                img_name = (
                    f"{count + 1:03d}.jpg"
                )

                img_path = os.path.join(
                    folder_name,
                    img_name
                )

                # ----------------------
                # HTTP IMAGE
                # ----------------------
                if img_url.startswith(
                    "http"
                ):

                    response = requests.get(
                        img_url,
                        timeout=10
                    )

                    if (
                        response.status_code
                        == 200
                    ):

                        with open(
                            img_path,
                            "wb"
                        ) as f:

                            f.write(
                                response.content
                            )

                        count += 1

                # ----------------------
                # BASE64 IMAGE
                # ----------------------
                elif img_url.startswith(
                    "data:image"
                ):

                    img_data = (
                        img_url.split(
                            "base64,"
                        )[1]
                    )

                    image = Image.open(
                        io.BytesIO(
                            base64.b64decode(
                                img_data
                            )
                        )
                    )

                    image.save(
                        img_path
                    )

                    count += 1

            except Exception as e:
                print(
                    f"Skipped image: {e}"
                )

        print(
            f"Downloaded "
            f"{count} images"
        )

    # ==========================
    # CLOSE BROWSER
    # ==========================
    driver.quit()

    print("\nDone!")


# ==========================
# EXAMPLE USAGE
# ==========================
queries = [
    "imran khan",
    "nawaz sharif",
    "maryam nawaz",
    "shehbaz sharif",
    "bilawal bhutto zardari",
    "benazir bhutto",
    "asif ali zardari",
    "altaf hussain",
    "fazlur rehman",
    "pervez musharraf",
    "chaudhry shujaat hussain",
    "mohsin naqvi",
    "mohsin dawar",
    "siraj ul haq",
    "mustafa kamal",
    "asif ghafoor"
]

download_google_images(queries)
