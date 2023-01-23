pub mod build_matrix;
pub mod parse_sam;
mod rewrite_align;

use pyo3::prelude::*;
use std::collections::HashMap;

#[pymodule]
///pyo3 interface
fn virtool_expectation_maximization(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    return Ok(());
}

#[pyfunction]
///Entry point for the virtool_expectation_maximization python module
pub fn run(
    _py: Python,
    sam_path: String,
    reassigned_path: String,
    p_score_cutoff: f64,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<String>,
    Vec<String>,
) {
    let (u, nu, refs, reads) = build_matrix::build_matrix(sam_path.as_str(), None);

    let (best_hit_initial_reads, best_hit_initial, level_1_initial, level_2_initial) =
        compute_best_hit(&u, &nu, &refs, &reads);

    let (init_pi, pi, _, nu) = em(&u, nu, &refs, 50, 1e-7, 0.0, 0.0);

    let (best_hit_final_reads, best_hit_final, level_1_final, level_2_final) =
        compute_best_hit(&u, &nu, &refs, &reads);

    rewrite_align::rewrite_align(
        &u,
        &nu,
        sam_path.as_str(),
        &p_score_cutoff,
        &reassigned_path,
    );

    return (
        best_hit_initial_reads,
        best_hit_initial,
        level_1_initial,
        level_2_initial,
        best_hit_final_reads,
        best_hit_final,
        level_1_final,
        level_2_final,
        init_pi,
        pi,
        refs,
        reads,
    );
}

pub fn compute_best_hit(
    u: &HashMap<i32, (i32, f64)>,
    nu: &HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)>,
    refs: &Vec<String>,
    reads: &Vec<String>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let ref_count = refs.len();
    let mut best_hit_reads = vec![0.0; ref_count];
    let mut level_1_reads = vec![0.0; ref_count];
    let mut level_2_reads = vec![0.0; ref_count];

    for i in u.keys() {
        *(best_hit_reads
            .get_mut(u.get(i).unwrap().0 as usize)
            .unwrap()) += 1.0;
        *(level_1_reads.get_mut(u.get(i).unwrap().0 as usize).unwrap()) += 1.0;
    }

    for i in nu.keys() {
        let z = nu.get(i).unwrap();
        let ind = &z.0;
        let x_norm = &z.2;
        let best_ref = x_norm.iter().cloned().fold(-1. / 0. /* -inf */, f64::max);
        let mut num_best_ref = 0;

        for (j, _) in x_norm.iter().enumerate() {
            if *(x_norm.get(j).unwrap()) == best_ref {
                num_best_ref += 1;
            }
        }

        num_best_ref = match num_best_ref {
            0 => 1,
            _ => num_best_ref,
        };

        for (j, _) in x_norm.iter().enumerate() {
            if *(x_norm.get(j).unwrap()) == best_ref {
                *(best_hit_reads
                    .get_mut(*(ind.get(j).unwrap()) as usize)
                    .unwrap()) += 1.0 / num_best_ref as f64;

                if *(x_norm.get(j).unwrap()) >= 0.5 {
                    *(level_1_reads
                        .get_mut(*(ind.get(j).unwrap()) as usize)
                        .unwrap()) += 1.0;
                } else if *(x_norm.get(j).unwrap()) >= 0.01 {
                    *(level_2_reads
                        .get_mut(*(ind.get(j).unwrap()) as usize)
                        .unwrap()) += 1.0;
                }
            }
        }
    }

    let read_count = reads.len();

    let best_hit: Vec<f64> = best_hit_reads
        .iter()
        .map(|val| val.clone() / read_count as f64)
        .collect();
    let level1: Vec<f64> = level_1_reads
        .iter()
        .map(|val| val.clone() / read_count as f64)
        .collect();
    let level2: Vec<f64> = level_2_reads
        .iter()
        .map(|val| val.clone() / read_count as f64)
        .collect();

    return (best_hit_reads, best_hit, level1, level2);
}

pub fn em(
    u: &HashMap<i32, (i32, f64)>,
    mut nu: HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)>,
    genomes: &Vec<String>,
    max_iter: i32,
    epsilon: f64,
    pi_prior: f64,
    theta_prior: f64,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    HashMap<i32, (Vec<i32>, Vec<f64>, Vec<f64>, f64)>,
) {
    let genome_count = genomes.len();
    let mut pi = vec![1.0 as f64 / genome_count as f64; genome_count];
    let mut init_pi = pi.clone();
    let mut theta = pi.clone();

    let mut pi_sum_0 = vec![0.0; genome_count];

    let u_weights: Vec<f64> = u.iter().map(|entry| (*(entry.1)).1).collect();
    let mut max_u_weights = 0.0;
    let mut u_total = 0.0;

    if !u_weights.is_empty() {
        max_u_weights = u_weights
            .iter()
            .cloned()
            .fold(-1. / 0. /* -inf */, f64::max);
        u_total = u_weights.iter().sum();
    }

    for i in u.keys() {
        pi_sum_0[u.get(i).unwrap().0 as usize] += u.get(i).unwrap().1;
    }

    let nu_weights: Vec<f64> = nu.iter().map(|entry| (*(entry.1)).3).collect();
    let mut max_nu_weights = 0.0;
    let mut nu_total = 0.0;

    if !nu_weights.is_empty() {
        max_nu_weights = nu_weights
            .iter()
            .cloned()
            .fold(-1. / 0. /* -inf */, f64::max);
        nu_total = nu_weights.iter().sum();
    }

    let prior_weight = f64::max(max_u_weights, max_nu_weights);
    let mut nu_length = nu.len();

    if nu_length == 0 {
        nu_length = 1;
    }

    //EM iterations
    for i in 0..max_iter {
        let pi_old = pi.clone();
        let mut theta_sum = vec![0.0; genome_count];

        //E step
        for j in nu.clone().keys() {
            let z = nu.get(j).unwrap().clone();

            //A set of any genome mapping with j
            let ind = &z.0;

            //Get relevant pis for the read
            let pi_temp: Vec<f64> = ind.iter().map(|val| pi[*val as usize].clone()).collect();

            //Get relevant thetas for the read
            let theta_temp: Vec<f64> = ind.iter().map(|val| theta[*val as usize].clone()).collect();

            //Calculate non-normalized xs
            let mut x_temp: Vec<f64> = Vec::new();

            for k in 0..ind.len() {
                x_temp.push(pi_temp[k] * theta_temp[k] * z.1[k]);
            }

            let x_sum: f64 = x_temp.iter().sum();

            //Avoid dividing by 0 at all times
            let x_norm: Vec<f64>;

            if x_sum == 0.0 {
                x_norm = vec![0.0; x_temp.len()];
            } else {
                x_norm = x_temp.iter().map(|val| val / x_sum).collect();
            }

            //Update x in nu
            nu.get_mut(j).unwrap().2 = x_norm.clone();

            for (k, _) in ind.iter().enumerate() {
                theta_sum[ind[k] as usize] += x_norm[k] * nu.get(j).unwrap().3;
            }
        }

        //M step
        let pi_sum: Vec<f64> = theta_sum
            .iter()
            .enumerate()
            .map(|(idx, _)| theta_sum[idx] + pi_sum_0[idx])
            .collect();
        let pip = pi_prior * prior_weight;

        //update pi
        pi = pi_sum
            .iter()
            .map(|val| ((*val as f64) + pip) / (u_total + nu_total + (pip * pi_sum.len() as f64)))
            .collect();

        if i == 0 {
            init_pi = pi.clone();
        }

        let theta_p = theta_prior * prior_weight;

        let mut nu_total_div = nu_total;

        if nu_total_div == 0 as f64 {
            nu_total_div = 1 as f64;
        }

        theta = theta_sum
            .iter()
            .map(|val| (*val + theta_p) / (nu_total_div + (theta_p * theta_sum.len() as f64)))
            .collect();

        let mut cutoff = 0.0;

        for (k, _) in pi.iter().enumerate() {
            cutoff += (pi_old[k] - pi[k]).abs();
        }

        if cutoff <= epsilon || nu_length == 1 {
            break;
        }
    }

    return (init_pi, pi, theta, nu);
}

///tests and whatnot
#[cfg(test)]
mod tests {

    #![allow(unused)]

    use crate::build_matrix::*;
    use crate::rewrite_align::*;
    use crate::*;
    use std::fs::File;
    use std::io::BufRead;
    use std::io::BufReader;
    use std::io::Read;

    extern crate yaml_rust;
    use yaml_rust::{YamlEmitter, YamlLoader};

    #[test]
    fn test_rewrite_align() {
        let (u, nu, refs, reads) = build_matrix("TestFiles/test_al.sam", None);
        let (init_pi, pi, theta, nu) = em(&u, nu, &refs, 5, 1e-7, 0.0, 0.0);
        rewrite_align(
            &u,
            &nu,
            "TestFiles/test_al.sam",
            &0.01,
            "TestFiles/rewrite.sam",
        );

        let mut new_file =
            BufReader::new(File::open("TestFiles/rewrite.sam").expect("Invalid file"));
        let mut test_file = BufReader::new(
            File::open("tests/test_pathoscope/test_rewrite_align.txt").expect("Invalid file"),
        );

        let mut new_line = String::new();
        let mut test_line = String::new();

        //compare output
        loop {
            new_line.clear();
            test_line.clear();

            match new_file.read_line(&mut new_line) {
                Ok(val) => {
                    if val == 0 {
                        return;
                    }
                }

                Err(_) => panic!("tests::test_rewrite_align: `error reading rewrite.sam`"),
            }

            match test_file.read_line(&mut test_line) {
                Ok(val) => {
                    if val == 0 {
                        return;
                    }
                }

                Err(_) => {
                    panic!("tests::test_rewrite_align `error reading test_rewrite_align.txt`")
                }
            }

            let new_line_fields: Vec<&str> = new_line.split('\t').collect();
            let test_line_fields: Vec<&str> = test_line.split('\t').collect();

            //compare fields of desired output to actual output
            for i in 0..new_line_fields.len() {
                assert!(
                    new_line_fields[i] == test_line_fields[i],
                    "tests::test_rewrite_align: `output of rewrite_align does not match expected`"
                );
            }
        }
    }

    #[test]
    fn test_em() {
        let (u, nu, refs, reads) = build_matrix("TestFiles/test_al.sam", None);
        let (init_pi, pi, theta, nu) = em(&u, nu, &refs, 5, 1e-6, 0.0, 0.0);

        let mut test_file = File::open("tests/test_pathoscope/test_em_5_1e_06_0_0_.yml")
            .expect("tests::test_build_matrix: `unable to open test file`");
        let mut test_string = String::new();
        test_string.clear();
        test_file
            .read_to_string(&mut test_string)
            .expect("tests::test_build_matrix: unable to read test file");
        let test_matrix = YamlLoader::load_from_str(&test_string)
            .expect("tests::test_build_matrix: unable to parse test file as .yml")[0]
            .clone();

        //compare nu
        for (key, value) in nu {
            for i in 0..value.0.len() {
                assert!(value.0[i] as i64 == test_matrix[3][key as usize][0][i].as_i64().unwrap());
            }
        }

        let mut init_pi_count = 0;

        //compare init_pi
        for entry in &init_pi {
            //vector is not sorted; check every index and break if found
            for i in 0..init_pi.len() {
                if ((*entry) - test_matrix[0][i].as_f64().unwrap()) <= 0.000000000000001 {
                    init_pi_count += 1;
                    break;
                } else {
                    continue;
                }
            }
        }
        if init_pi_count != test_matrix[0].as_vec().unwrap().len() {
            panic!();
        }

        let mut pi_count = 0;

        //compare pi
        for entry in &pi {
            //vector is not sorted; check every index and break if found
            for i in 0..pi.len() {
                if ((*entry) - test_matrix[1][i].as_f64().unwrap()) <= 0.000000000000001 {
                    pi_count += 1;
                    break;
                } else {
                    continue;
                }
            }
        }
        if pi_count != test_matrix[1].as_vec().unwrap().len() {
            panic!();
        }

        let mut theta_count = 0;

        //compare pi
        for entry in &theta {
            //vector is not sorted; check every index and break if found
            for i in 0..theta.len() {
                if ((*entry) - test_matrix[2][i].as_f64().unwrap()) <= 0.000000000000001 {
                    theta_count += 1;
                    break;
                } else {
                    continue;
                }
            }
        }
        if theta_count != test_matrix[2].as_vec().unwrap().len() {
            panic!();
        }
    }

    #[test]
    fn test_best_hit() {
        let (u, nu, refs, reads) = build_matrix("TestFiles/test_al.sam", None);
        let (best_hit_reads, best_hit, level1, level2) = compute_best_hit(&u, &nu, &refs, &reads);

        let mut test_file = File::open("tests/test_pathoscope/test_compute_best_hit.yml")
            .expect("tests::test_build_matrix: `unable to open test file`");
        let mut test_string = String::new();
        test_string.clear();
        test_file
            .read_to_string(&mut test_string)
            .expect("tests::test_build_matrix: unable to read test file");
        let test_matrix = YamlLoader::load_from_str(&test_string)
            .expect("tests::test_build_matrix: unable to parse test file as .yml")[0]
            .clone();

        //compare best_hit_reads
        for i in 0..best_hit_reads.len() {
            assert!(best_hit_reads[i] == test_matrix[0][i].as_f64().unwrap())
        }

        //compare best_hit
        for i in 0..best_hit.len() {
            assert!(best_hit[i] == test_matrix[1][i].as_f64().unwrap());
        }

        //compare level1
        for i in 0..level1.len() {
            assert!(level1[i] == test_matrix[2][i].as_f64().unwrap());
        }

        //compare level2
        for i in 0..level2.len() {
            assert!(level2[i] == test_matrix[3][i].as_f64().unwrap());
        }
    }

    #[test]
    fn test_build_matrix() {
        let (u, nu, refs, reads) = build_matrix("TestFiles/test_al.sam", None);

        let mut test_file = File::open("tests/test_pathoscope/test_build_matrix.yml")
            .expect("tests::test_build_matrix: `unable to open test file`");
        let mut test_string = String::new();
        test_string.clear();
        test_file
            .read_to_string(&mut test_string)
            .expect("tests::test_build_matrix: unable to read test file");
        let test_matrix = YamlLoader::load_from_str(&test_string)
            .expect("tests::test_build_matrix: unable to parse test file as .yml")[0]
            .clone();

        //compare u
        for (key, value) in u {
            assert!(value.0 as i64 == test_matrix[0][key as usize][0].as_i64().unwrap());
            assert!(value.1 == test_matrix[0][key as usize][1].as_f64().unwrap());
        }

        //compare nu
        for (key, value) in nu {
            for i in 0..value.0.len() {
                assert!(value.0[i] as i64 == test_matrix[1][key as usize][0][i].as_i64().unwrap());
            }

            for i in 0..value.1.len() {
                assert!(value.1[i] == test_matrix[1][key as usize][1][i].as_f64().unwrap());
            }

            for i in 0..value.2.len() {
                assert!(value.2[i] == test_matrix[1][key as usize][2][i].as_f64().unwrap());
            }

            assert!(value.3 == test_matrix[1][key as usize][3].as_f64().unwrap());
        }

        let mut ref_count = 0;

        //compare refs
        for entry in &refs {
            //vector is not sorted; check every index and break if found
            for i in 0..refs.len() {
                if (*entry).eq(test_matrix[2][i].as_str().unwrap()) {
                    ref_count += 1;
                    break;
                } else {
                    continue;
                }
            }
        }
        if ref_count != test_matrix[2].as_vec().unwrap().len() {
            panic!();
        }

        let mut read_count = 0;

        //compare reads
        for entry in &reads {
            //vector is not sorted; check every index and break if found
            for i in 0..reads.len() {
                if (*entry).eq(test_matrix[3][i].as_str().unwrap()) {
                    read_count += 1;
                    break;
                } else {
                    continue;
                }
            }
        }
        if read_count != test_matrix[3].as_vec().unwrap().len() {
            panic!();
        }
    }
}
