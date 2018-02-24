#/bin/bash

#DATA_DIR=/home/liquida/DataAnalysis/gender/GenderClass/data
usage()
{
cat << EOF
usage: $0 options

This script run a partial do all for gender analisys  .

OPTIONS:
   -h      Show this message
   -d      Path for the data dir in gender repo
   -m      Mod File, aka DA_SETTINGS*
   -f      Force to run on a specific date
   -k      Krux file generated from the datalake
EOF
}

DATA_DIR=
DA_SETTINGS=
FORCE_DATE=
KRUX_FILE=

while getopts “hd:m:f:k:” OPTION
do
     case $OPTION in
         h)
             usage
             exit 1
             ;;
         d)
             DATA_DIR=$OPTARG
             ;;
         m)
             DA_SETTINGS=$OPTARG
             ;;
         f)
             FORCE_DATE=$OPTARG
             ;;
         k)
             KRUX_FILE=$OPTARG
             ;;
         ?)
             usage
             exit
             ;;
     esac
done

if [[ -z $DATA_DIR ]] || [[ -z $DA_SETTINGS ]]
then
     usage
     exit 1
fi
 #Esempio di gestione degli input con bash.
 
