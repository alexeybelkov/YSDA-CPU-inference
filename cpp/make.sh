scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "$scriptDir" || exit 1
mkdir build
cd build
rm -rf *
cmake ..
make
