files=./*.pdf
for file in $files; do
pdfcrop $file $file
done

