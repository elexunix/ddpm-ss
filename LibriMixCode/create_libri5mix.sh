#!/bin/bash
set -eu  # Exit on error

storage_dir=data/datasets/librimix-10

if [ ! -d metadata/Libri5Mix ]; then
  echo GENERATING METADATA FOR 5 SPEAKERS
  python3 scripts/create_librimix_metadata.py --librispeech_dir $storage_dir/LibriSpeech --librispeech_md_dir metadata/LibriSpeech --wham_dir $storage_dir/wham_noise --wham_md_dir metadata/Wham_noise --metadata metadata/Libri5Mix --n_src 5
  echo TRUNCATING TABLES
  sed -i '10002,$d' metadata/Libri5Mix/*
fi

echo GENERATED METADATA SUCCESSFULLY

# If you wish to rerun this script in the future please comment this line out.
python3 scripts/augment_train_noise.py --wham_dir $storage_dir/wham_noise

for n_src in 5; do
  metadata_dir=metadata/Libri$n_src"Mix"
  python3 scripts/create_librimix_from_metadata.py --librispeech_dir $storage_dir/LibriSpeech \
    --wham_dir $storage_dir/wham_noise \
    --metadata_dir $metadata_dir \
    --librimix_outdir $storage_dir/ \
    --n_src $n_src \
    --freqs 16k \
    --modes max \
    --types mix_clean mix_both
done
#    --freqs 8k 16k \
#    --modes min max \
#    --types mix_clean mix_both mix_single
