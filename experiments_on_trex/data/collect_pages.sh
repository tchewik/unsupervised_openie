if [ ! -f "$1" ]; then
    printf 'Download %s ...\n' "$1"
    wget http://download.wikimedia.org/enwiki/latest/$1
fi

git clone https://github.com/tchewik/wikiextractor.git

python wikiextractor/WikiExtractor.py $1 -q \
       --json \
       --processes 8 \
       --output extracted \
       --bytes 8M \
       --compress \
       --filter_category $2 \
       --min_text_length 1
       
find extracted -name '*bz2' -exec bzip2 -dkc {} \; > $3

rm -r wikiextractor