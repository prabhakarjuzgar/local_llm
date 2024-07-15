import os
from typing import Tuple
import functools
import boto3
import awswrangler as wrangler


@functools.cache
async def get_boto3_session() -> boto3.session.Session:
    return boto3.session.Session()


@functools.cache
async def get_bucket_name() -> str:
    return os.environ.get("BUCKET_NAME", "")


@functools.cache
async def get_path(bucket_name:str, file: str) -> str:
    bucket_path = os.environ.get("BUCKET_PATH", "")
    return f"s3://{bucket_name}/{bucket_path}{file}"


async def upload_to_s3(file) -> Tuple[str, str]:
    bucket_name = await get_bucket_name()
    if not bucket_name:
        raise ModuleNotFoundError

    file_name = file.file_name.split("/")[-1]
    s3_file_path = await get_path(bucket_name, file_name)
    wrangler.s3.upload(
        local_file=file.file_name,
        path=get_path(s3_file_path),
        boto3_session=get_boto3_session()
    )

    return file_name, s3_file_path


async def download_from_s3(file):
    bucket_name = await get_bucket_name()
    if not bucket_name:
        raise ModuleNotFoundError

    file_name = file.file_name.split("/")[-1]
    s3_file_path = await get_path(bucket_name, file_name)
    wrangler.s3.download(
        path=s3_file_path,
        local_file=file,
        boto3_session=get_boto3_session()
    )

