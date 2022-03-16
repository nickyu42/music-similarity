#!/usr/bin/env fish

if test (count $argv) -ne 2
    echo "Usage:" (status filename) "[source] [audio out]"
    exit 1
end

set LINKS_SOURCE $argv[1]
set WAV_DIR $argv[2]

mkdir -p $WAV_DIR

for link in (xsv select "link" $LINKS_SOURCE)

    # Get file name from URL, which is the last part of the path
    set filename (string split "/" $link)[-1]

    # Change extension to wave file by splitting at last dot
    # and addding extension
    set audio_filename (string split -r -m 1 "." $filename)[1].wav

    wget \
        # Skip files that are already downloaded
        --no-clobber \
        # Set recursision with 1 level depth
        -r -l1 \
        -U "Mozilla/5.0 (X11; Linux i686 (x86_64)) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.75 Safari/537.36" \
        $link

    ffmpeg -n -i animethemes.moe/video/$filename $WAV_DIR/$audio_filename
end