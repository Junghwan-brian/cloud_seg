from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="torchgeo/l8biome",     # Hugging Face 데이터셋 ID
    repo_type="dataset",            # dataset으로 지정
    local_dir="/home/telepix_nas/junghwan/cloud_seg/l8biome",     # 로컬 저장 폴더
    local_dir_use_symlinks=False    # 실제 파일로 저장
)
