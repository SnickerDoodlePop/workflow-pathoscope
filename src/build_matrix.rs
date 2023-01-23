use std::{collections::HashMap, fs::File, io::BufReader};

use crate::parse_sam::*;

pub fn build_matrix(
    sam_path: &str,
    p_score_cutoff: Option<f64>,
) -> (
    HashMap<i32, (i32, f64)>,
    HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)>,
    Vec<String>,
    Vec<String>,
) {
    let mut u: HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)> = HashMap::new();
    let mut nu: HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)> = HashMap::new();

    let mut h_read_id: HashMap<String, i32> = HashMap::new();
    let mut h_ref_id: HashMap<String, i32> = HashMap::new();

    let mut refs: Vec<String> = Vec::new();
    let mut reads: Vec<String> = Vec::new();

    let mut ref_count: i32 = 0;
    let mut read_count: i32 = 0;

    let mut max_score: f64 = 0.0;
    let mut min_score: f64 = 0.0;

    let sam_file = File::open(sam_path).expect("Invalid file");
    let mut sam_reader = BufReader::new(sam_file);

    loop {
        let new_line: SamLine;

        match parse_sam(&mut sam_reader, p_score_cutoff) {
            ParseResult::Ok(data) => {
                if data.score < p_score_cutoff {
                    continue;
                } else {
                    new_line = data;
                }
            }
            ParseResult::EOF => break,
            ParseResult::Err(msg) => {
                println!("{}", msg);
                panic!();
            }
            ParseResult::Ignore => continue,
        }

        min_score = new_line.score.unwrap().min(min_score);
        max_score = new_line.score.unwrap().max(max_score);

        let mut ref_index = *(h_ref_id.get(&new_line.ref_id).unwrap_or(&-1));

        if ref_index == -1 {
            ref_index = ref_count;
            h_ref_id.insert(new_line.ref_id.clone(), ref_index);
            refs.push(new_line.ref_id.clone());
            ref_count += 1;
        }

        let mut read_index = *(h_read_id.get(&new_line.read_id).unwrap_or(&-1));

        if read_index == -1 {
            read_index = read_count;
            h_read_id.insert(new_line.read_id.clone(), read_index);
            reads.push(new_line.read_id.clone());
            read_count += 1;

            u.insert(
                read_index,
                (
                    vec![ref_index],
                    vec![new_line.score.clone().unwrap()],
                    vec![new_line.score.clone().unwrap() as f64],
                    new_line.score.clone().unwrap(),
                ),
            );
        } else {
            if u.contains_key(&read_index) {
                if u.get(&read_index).unwrap().0.contains(&ref_index) {
                    continue;
                }
                nu.insert(read_index, u.get(&read_index).unwrap().clone());
                u.remove(&read_index);
            }

            if nu.get(&read_index).unwrap().0.contains(&ref_index) {
                continue;
            }

            nu.get_mut(&read_index).unwrap().0.push(ref_index);
            nu.get_mut(&read_index)
                .unwrap()
                .1
                .push(new_line.score.unwrap());

            if new_line.score.unwrap() > nu.get(&read_index).unwrap().3 {
                nu.get_mut(&read_index).unwrap().3 = new_line.score.unwrap();
            }
        }
    }

    let (u, mut nu) = rescale_samscore(u, nu, max_score, min_score);

    let mut u_return: HashMap<i32, (i32, f64)> = HashMap::new();

    for k in u.keys() {
        u_return.insert(
            *k,
            (
                u.get(k).unwrap().0.get(0).unwrap().clone(),
                u.get(k).unwrap().1.get(0).unwrap().clone(),
            ),
        );
    }

    for k in nu.clone().keys() {
        let p_score_sum = nu.get(k).unwrap().1.iter().sum::<f64>();

        nu.get_mut(k).unwrap().2 = nu
            .get(k)
            .unwrap()
            .1
            .iter()
            .map(|data| data / p_score_sum)
            .collect();
    }

    return (u_return, nu, refs, reads);
}

///modifies the scores of u and nu with respect to max_score and min_score
fn rescale_samscore(
    mut u: HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)>,
    mut nu: HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)>,
    max_score: f64,
    min_score: f64,
) -> (
    HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)>,
    HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)>,
) {
    let scaling_factor: f64;

    if min_score < 0.0 {
        scaling_factor = 100.0 / (max_score - min_score);
    } else {
        scaling_factor = 100.0 / max_score;
    }

    for k in u.clone().keys() {
        if min_score < 0.0 {
            u.get_mut(k).unwrap().1[0] = u.get(k).unwrap().1[0].clone() - min_score;
        }

        u.get_mut(k).unwrap().1[0] = f64::exp(u.get(k).unwrap().1[0] * scaling_factor);
        u.get_mut(k).unwrap().3 = u.get(k).unwrap().1[0];
    }

    for k in nu.clone().keys() {
        nu.get_mut(k).unwrap().3 = 0.0;

        for i in 0..nu.get(k).unwrap().1.len() {
            if min_score < 0.0 {
                nu.get_mut(k).unwrap().1[i] = nu.get(k).unwrap().1[i] - min_score;
            }

            nu.get_mut(k).unwrap().1[i] = f64::exp(nu.get(k).unwrap().1[i] * scaling_factor);

            if nu.get(k).unwrap().1[i] > nu.get(k).unwrap().3 {
                nu.get_mut(k).unwrap().3 = nu.get(k).unwrap().1[i];
            }
        }
    }
    return (u, nu);
}
