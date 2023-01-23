use std::{
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader},
};

///Stores the desired fields of a .SAM record and the line itself as a String
#[derive(Debug)]
pub struct SamLine {
    pub read_id: String,
    pub read_length: usize,
    pub position: u32,
    pub score: Option<f64>,
    pub btws_flg: u32,
    pub unmapped: bool,
    pub ref_id: String,
    pub sam_fields: Vec<String>,
    pub line: String,
}

impl SamLine {
    ///Create a new Some(SamLine) object by consuming a String object
    ///
    ///Returns none if the provided String is to be ignored
    pub fn new(new_line: String) -> Option<SamLine> {
        if new_line.is_empty() || new_line.starts_with("#") || new_line.starts_with("@") {
            return None;
        }

        let fields = new_line.split("\t").collect::<Vec<&str>>();

        //extremely inefficient; should optimize later on
        let mut new_sam_line = SamLine {
            read_id: String::from(*(fields.get(0).expect("error parsing read_id"))),
            read_length: fields.get(9).expect("error parsing length field").len(),
            position: fields
                .get(3)
                .expect("error reading position field")
                .parse::<u32>()
                .expect("error parsing position as u32"),
            score: None,
            btws_flg: fields
                .get(1)
                .expect("error reading btws_flg field")
                .parse::<u32>()
                .expect("error parsing btws_flg as u32"),
            unmapped: ((fields.get(1).unwrap().parse::<u32>().unwrap()) & (4 as u32) == (4 as u32)),
            ref_id: String::from(*(fields.get(2).expect("error parsing ref_id"))),
            sam_fields: fields
                .into_iter()
                .map(|data| String::from(data))
                .collect::<Vec<String>>(),
            line: new_line,
        };

        new_sam_line.score = Some(find_sam_align_score(&mut new_sam_line));

        return Some(new_sam_line);
    }
}

fn find_sam_align_score(data: &SamLine) -> f64 {
    for field in data.sam_fields.clone() {
        if field.starts_with("AS:i:") {
            return (field[5..]
                .parse::<i32>()
                .expect("unable to parse field as i32 in find_sam_align_score")
                as f64)
                + (data.read_length as f64);
        }
    }

    panic!("unable to find sam alignment score!")
}

/// stores the result of parsing one line of a .SAM file\
/// * Ok(T) => T is a SamLine object;\
/// * Ignore and EOF are special flags;\
/// * Err(String) => String indicates an error generating a SamLine object.
pub enum ParseResult<T> {
    Ok(T),
    Ignore,
    EOF,
    Err(String),
}

pub fn parse_sam(
    sam_reader: &mut BufReader<File>,
    p_score_cutoff: Option<f64>,
) -> ParseResult<SamLine> {
    let p_score_cutoff: f64 = p_score_cutoff.unwrap_or(0.01);

    let mut buf: String = String::new();
    match sam_reader.read_line(&mut buf) {
        Ok(code) => {
            if code == 0 {
                return ParseResult::EOF;
            } else {
                match SamLine::new(buf) {
                    None => return ParseResult::Ignore,
                    Some(new_line) => {
                        if new_line
                            .score
                            .expect("error unwrapping newline.score in parseSAM")
                            > p_score_cutoff
                        {
                            return ParseResult::Ok(new_line);
                        } else {
                            return ParseResult::Ignore;
                        }
                    }
                }
            }
        }
        Err(_) => {
            return ParseResult::Err(String::from(
                "Error propagated in parseSAM from SamLine::new",
            ))
        }
    }
}
