#!/bin/bash
set -eo pipefail

src_utt2duration="data/librilight-6000h/3_65_utt2duration"

dst1_dir="data/librilight-100h"
dst2_dir="data/librilight-100h-400h"
dst3_dir="data/librilight-100h-400h-500h"
dst4_dir="data/librilight-100h-400h-500h-1000h"

for dir in ${dst2_dir} ${dst3_dir} ${dst4_dir}; do
    cat "data/librilight-6000h/3_65_wav.scp" | grep -Ff ${dir}/uttids > ${dir}/wav.scp
done

exit
echo "warning"

for dir in ${dst2_dir} ${dst3_dir} ${dst4_dir}; do
    for file in uttids utt2duration; do
        if [ -f ${dir}/${file} ]; then rm ${dir}/${file}; fi
    done
done


function select_uttids() {
    var_utt2dur=$1
    var_taregt_durations=$2
    var_uttid_out=$3
    var_min=30
    var_max=65
    cat ${var_utt2dur} | shuf  | awk -v min=${var_min} -v max=${var_max} -v target=${var_taregt_durations} 'BEGIN{S=0;}{
        if($2>=min&&$2<=max&&S<target) {
            S+=$2;
            print $1;
        }
    }' > ${var_uttid_out}
}

tmpdir=$(mktemp -d tmp.XXX)

# filter out target1
cat ${src_utt2duration} | grep -Fvf ${dst1_dir}/uttids > ${tmpdir}/utt2duration_for_dst2
# select for dst2, 400h
select_uttids ${tmpdir}/utt2duration_for_dst2 1440000 ${tmpdir}/uttids_400h
cat ${dst1_dir}/uttids ${tmpdir}/uttids_400h > ${dst2_dir}/uttids
# filter utt2duration, count total
cat ${src_utt2duration} | grep -Ff ${dst2_dir}/uttids > ${dst2_dir}/utt2duration
echo "${dst2_dir}: total: $(cat ${dst2_dir}/utt2duration | awk '{S+=$2}END{print S/3600}')h"

# filter out target2
cat ${src_utt2duration} | grep -Fvf ${dst2_dir}/uttids > ${tmpdir}/utt2duration_for_dst3
# select for dst3, 500h
select_uttids ${tmpdir}/utt2duration_for_dst3 1800000 ${tmpdir}/uttids_500h
cat ${dst2_dir}/uttids ${tmpdir}/uttids_500h > ${dst3_dir}/uttids
# filter utt2duration, count total
cat ${src_utt2duration} | grep -Ff ${dst3_dir}/uttids > ${dst3_dir}/utt2duration
echo "${dst3_dir}: total: $(cat ${dst3_dir}/utt2duration | awk '{S+=$2}END{print S/3600}')h"

# filter out target3
cat ${src_utt2duration} | grep -Fvf ${dst3_dir}/uttids > ${tmpdir}/utt2duration_for_dst4
# select for dst4, 1000h
select_uttids ${tmpdir}/utt2duration_for_dst4 3600000 ${tmpdir}/uttids_1000h
cat ${dst3_dir}/uttids ${tmpdir}/uttids_1000h > ${dst4_dir}/uttids
# filter utt2duration, count total
cat ${src_utt2duration} | grep -Ff ${dst4_dir}/uttids > ${dst4_dir}/utt2duration
echo "${dst4_dir}: total: $(cat ${dst4_dir}/utt2duration | awk '{S+=$2}END{print S/3600}')h"


rm -r ${tmpdir}
