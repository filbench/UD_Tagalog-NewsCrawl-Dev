branch="dev"

while getopts b: flag
do
    case "${flag}" in
        b) branch=${OPTARG};;
    esac
done
echo "Downloading files from the source..."
mkdir -p assets
wget -O assets/tl_newscrawl-ud-train.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Tagalog-NewsCrawl/refs/heads/$branch/tl_newscrawl-ud-train.conllu
wget -O assets/tl_newscrawl-ud-dev.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Tagalog-NewsCrawl/refs/heads/$branch/tl_newscrawl-ud-dev.conllu
wget -O assets/tl_newscrawl-ud-test.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Tagalog-NewsCrawl/refs/heads/$branch/tl_newscrawl-ud-test.conllu