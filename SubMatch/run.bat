mkdir output
cd output
mkdir linkedin
mkdir facebook
mkdir uspatent
mkdir dblp
cd ..

SubMatch.exe mode=2 data=data\uspatent.lg query=data\uspatent.q maxfreq=100 stats=output\uspatent

SubMatch.exe mode=2 data=data\dblp.lg query=data\dblp.q maxfreq=100 stats=output\dblp