import internetarchive as ia
import gdown 
from pathlib import Path

def download_with_proxy(identifier, files, save_dir, proxies=None):
    if proxies is None:
        proxies = {
            'http': 'http://localhost:9092',
            'https': 'http://localhost:9092'
        }

    session = ia.ArchiveSession()
    # session.proxies.update(proxies)

    # get item
    item = ia.get_item(
        identifier,
        archive_session=session,
    )
    item.download(
            files=files,
            verbose=True,
            ignore_existing=False,
            checksum=False,
            checksum_archive=False,
            destdir=save_dir,
            no_directory=False,
            retries=None,
            item_index=None,
            ignore_errors=False,
            on_the_fly=True,
            return_responses=False,
            no_change_timestamp=False,
            timeout=None,
        )
    

def download_from_drive(file_id, file_name, save_dir: Path):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    
    url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=1&confirm=t'
    gdown.download(url=url, output=(save_dir / file_name).absolute().as_posix())