use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::LineWriter;
use std::io::Write;

use crate::parse_sam::*;

pub fn rewrite_align(
    u: &HashMap<i32, (i32, f64)>,
    nu: &HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)>,
    sam_path: &str,
    p_score_cutoff: &f64,
    path: &str,
) {
    let mut read_id_dict: HashMap<String, i32> = HashMap::new();
    let mut ref_id_dict: HashMap<String, i32> = HashMap::new();

    let mut genomes: Vec<String> = Vec::new();
    let mut read: Vec<String> = Vec::new();

    let mut ref_count = 0;
    let mut read_count = 0;

    let old_file = File::open(sam_path).expect("Invalid file");
    let mut sam_reader = BufReader::new(old_file);
    let new_file = File::create(path).expect("unable to create file");
    let mut sam_writer = LineWriter::new(new_file);

    //for line in parseSam
    loop {
        let sam_line = parse_sam(&mut sam_reader, Some(*p_score_cutoff));

        let sam_line = match sam_line {
            ParseResult::Ok(line) => line,
            ParseResult::Ignore => continue,
            ParseResult::EOF => break,
            ParseResult::Err(_) => {
                panic!("unable to read old_file in rewrite_align::rewrite_align")
            }
        };

        let mut ref_index = ref_id_dict.get(&sam_line.ref_id).unwrap_or(&-1).clone();

        if ref_index == -1 {
            ref_index = ref_count.clone();
            ref_id_dict.insert(sam_line.ref_id.clone(), ref_index);
            genomes.push(sam_line.ref_id);
            ref_count += 1;
        }

        let mut read_index = *read_id_dict.get(&sam_line.read_id).unwrap_or(&-1);

        if read_index == -1 {
            // hold on to this new read
            // first, wrap previous read profile and see if any previous read has a
            // same profile with that!

            read_index = read_count;
            read_id_dict.insert(sam_line.read_id.clone(), read_index);
            read.push(sam_line.read_id);
            read_count += 1;

            if u.contains_key(&read_index) {
                sam_writer
                    .write(sam_line.line.as_bytes())
                    .expect("unable to write to new_file in rewrite_align::rewrite_align");
                continue;
            }
        }

        if nu.contains_key(&read_index) {
            if find_updated_score(&nu, read_index, ref_index) < *p_score_cutoff {
                continue;
            }
            sam_writer
                .write(sam_line.line.as_bytes())
                .expect("unable to write to new_file in rewrite_align::rewrite_align");
        }
    }
}

fn find_updated_score(
    nu: &HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)>,
    read_index: i32,
    ref_index: i32,
) -> f64 {
    let v1 = match nu.get(&read_index) {
        Some(val) => val,
        None => return 0.0,
    };

    let mut idx: usize = 0;

    for (i, el) in v1.0.iter().enumerate() {
        if *el == ref_index {
            idx = i;
            break;
        }
    }

    return *nu.get(&read_index).unwrap().2.get(idx).unwrap();
}
