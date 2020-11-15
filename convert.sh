for i in $1/*.mp3;
	do name=$(echo "$i" | rev | cut -d "/" -f 1 | rev |  cut -d "." -f 1)
	echo $name
	ffmpeg -i "$i" "$2/$name.wav" < /dev/null
done
