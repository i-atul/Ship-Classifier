import gdown

def download_from_gdrive(url: str, output: str = "products.zip"):
    """
    Download a file from Google Drive using gdown.
    Args:
        url (str): The Google Drive shareable link.
        output (str): The output filename.
    """
    # Extract the file ID from the URL
    drive_id = url.split('/')[-2]
    prefix = 'https://drive.google.com/uc?/export=download&id='
    gdown.download(prefix + drive_id, output)

# Example usage:
if __name__ == "__main__":
    url = "https://drive.google.com/file/d/1pqp5z5eJGh_USMSI5k9x8MSnWdAuOF9Z/view?usp=drive_link"
    download_from_gdrive(url)