import boto3
import sys


def get_commit(package, commit_hash):
    Bucket='ants-builds'
    s3 = boto3.client('s3')
    obj = s3.list_objects_v2(
            Bucket=Bucket,
            Prefix=f'{package}-builds/',
    )['Contents']
    keys = [i['Key'] for i in  obj]
    whl_files = [i for i in keys if '.whl' in i]
    commit_hashes = [i.split('/')[-2] for i in whl_files]
    if commit_hash not in commit_hashes:
        raise ValueError(f"Commit {commit_hash} not found")
    target = [i for i in whl_files if commit_hash in i][0]
    print(target)
    file_name = 'ext/' + target.split('/')[-1]
    s3.download_file(
            Bucket,
            target,
            file_name,
    )

if __name__=='__main__':
    package = sys.argv[1]
    commit_hash = sys.argv[2].replace('"', '')
    print((package, commit_hash))
    if package not in ['antspynet', 'antspy']:
        raise ValueError(f'Package should be antspy or antspynet')
    get_commit(package, commit_hash)
