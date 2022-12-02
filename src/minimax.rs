// Minimax implementation with heuristic functions is borrowed from
// https://github.com/ovolve/2048-AI

use itertools::Itertools;
use std::cmp::max;

use crate::state::*;

const SMOOTH_WEIGHT: f32 = 0.1;
const MONO_2_WEIGHT: f32 = 1.0;
const EMPTY_WEIGHT: f32 = 2.7;
const MAX_WEIGHT: f32 = 1.0;

const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

fn log_2(x: u64) -> u32 {
    assert!(x > 0);
    num_bits::<u64>() as u32 - x.leading_zeros() - 1
}

enum Turn {
    Player,
    Machine,
}

impl Turn {
    fn reverse(&self) -> Self {
        match self {
            Turn::Player => Turn::Machine,
            Turn::Machine => Turn::Player,
        }
    }
}

struct PlayerResult {
    best_score: f32,
    best_move: Option<Direction>,
}

pub struct MinimaxPlayer<'a> {
    state: &'a mut State,
}

impl<'a> MinimaxPlayer<'a> {
    pub fn new(state: &'a mut State) -> Self {
        MinimaxPlayer { state }
    }

    pub fn find_best_move(&mut self) -> Option<Direction> {
        let grid = &mut self.state.grid;
        let free_positions = grid.free_positions().count();
        let depth = match free_positions {
            0..=3 => 4,
            4..=7 => 3,
            8..=16 => 2,
            _ => unreachable!(),
        };
        grid.minimax_search(f32::NEG_INFINITY, f32::INFINITY, depth, &Turn::Player)
            .best_move
    }
}

impl Grid {
    fn minimax_search(&mut self, alpha: f32, beta: f32, depth: u16, turn: &Turn) -> PlayerResult {
        let mut best_score;
        let mut best_move = None;

        match turn {
            Turn::Player => {
                best_score = alpha;
                for &dir in &[
                    Direction::Up,
                    Direction::Left,
                    Direction::Down,
                    Direction::Right,
                ] {
                    let mut grid = self.clone();
                    if grid.update(&dir).success {
                        if depth == 0 {
                            return PlayerResult {
                                best_move: Some(dir),
                                best_score: grid.evaluate(),
                            };
                        }
                        let PlayerResult {
                            best_score: score, ..
                        } = grid.minimax_search(best_score, beta, depth - 1, &turn.reverse());
                        if score > best_score {
                            best_score = score;
                            best_move = Some(dir);
                        }
                        if best_score > beta {
                            // cutoff, quick return
                            return PlayerResult {
                                best_move,
                                best_score: beta,
                            };
                        }
                        // if there was any move whatsoever, return it
                        if best_move.is_none() {
                            best_move = Some(dir)
                        }
                    }
                }
            }
            Turn::Machine => {
                best_score = beta;

                struct Candidate {
                    x: usize,
                    y: usize,
                    value: u64,
                }
                let mut best_candidate_score = i32::MIN;

                let mut candidates = Vec::new();
                let positions: Vec<_> = self.free_positions().copied().collect();
                for (x, y) in positions {
                    for &value in &[2, 4] {
                        self.add(x, y, value);
                        let score = -self.smoothness() + self.num_islands() as i32;
                        if score > best_candidate_score {
                            candidates.clear();
                            best_candidate_score = score;
                        }
                        if score >= best_candidate_score {
                            candidates.push(Candidate { x, y, value });
                        }
                        self.remove(x, y);
                    }
                }
                for &Candidate { x, y, value } in &candidates {
                    let mut grid = self.clone();
                    grid.add(x, y, value);
                    let PlayerResult {
                        best_score: score, ..
                    } = grid.minimax_search(alpha, best_score, depth, &turn.reverse());

                    if score < best_score {
                        best_score = score;
                    }
                    if best_score < alpha {
                        return PlayerResult {
                            best_score: alpha,
                            best_move,
                        };
                    }
                }
            }
        }
        PlayerResult {
            best_score,
            best_move,
        }
    }

    fn evaluate(&self) -> f32 {
        self.smoothness() as f32 * SMOOTH_WEIGHT
            + self.monotonicity() as f32 * MONO_2_WEIGHT
            + (self.free_positions().count() as f32).ln() * EMPTY_WEIGHT
            + self.max_value().unwrap() as f32 * MAX_WEIGHT
    }

    fn free_positions(&self) -> impl Iterator<Item = &Position> {
        POSITIONS
            .iter()
            .filter(move |&&(x, y)| self[x][y].is_none())
    }

    fn find_next_position(&self, x: usize, y: usize, dir: &Direction) -> Option<Position> {
        let traversals = TRAVERSALS[*dir as usize];
        let traversal = match dir {
            Direction::Left | Direction::Right => traversals[x],
            Direction::Up | Direction::Down => traversals[y],
        };
        let mut start_search = false;
        for &(i, j) in traversal.iter().rev() {
            if self.contains(i, j) && start_search {
                return Some((i, j));
            }
            if (i, j) == (x, y) {
                start_search = true;
            }
        }
        None
    }

    fn max_value(&self) -> Option<u64> {
        POSITIONS
            .iter()
            .filter_map(|&(x, y)| self[x][y].as_ref())
            .map(|tile| tile.value)
            .max()
    }

    fn monotonicity(&self) -> i32 {
        let (mut up, mut down, mut left, mut right) = (0, 0, 0, 0);

        for i in 0..SIDE {
            macro_rules! calc_monotonicity {
                ($dir:expr, $var1:expr, $var2:expr) => {
                    let traversal = TRAVERSALS[$dir as usize][i];
                    for (p, q) in traversal
                        .iter()
                        .filter_map(|&(i, j)| {
                            let val = self[i][j]
                                .as_ref()
                                .map(|tile| log_2(tile.value))
                                .unwrap_or(0) as i32;
                            // If there are zeros, they only matter at
                            // the beginning of the line.
                            let idx = match $dir {
                                Direction::Down => i,
                                Direction::Right => j,
                                _ => unreachable!(),
                            };
                            if val == 0 && idx != 0 && idx != SIDE - 1 {
                                return None;
                            }
                            Some(val)
                        })
                        .tuple_windows()
                    {
                        if p > q {
                            $var1 += q - p;
                        } else {
                            $var2 += p - q;
                        }
                    }
                };
            }
            calc_monotonicity!(Direction::Down, up, down);
            calc_monotonicity!(Direction::Right, left, right);
        }
        max(up, down) + max(left, right)
    }

    // https://news.ycombinator.com/item?id=7381082
    // Find the sum of all edge weights in the new game state
    fn smoothness(&self) -> i32 {
        let mut smoothness = 0;
        self.tiles().for_each(|tile| {
            let (i, j) = tile.position;
            let p = log_2(tile.value) as i32;
            for dir in [Direction::Down, Direction::Right].iter() {
                if let Some((x, y)) = self.find_next_position(i, j, dir) {
                    if let Some(tile) = &self[x][y] {
                        let q = log_2(tile.value) as i32;
                        smoothness -= (p - q).abs();
                    }
                }
            }
        });
        smoothness
    }

    // Count the number of isolated groups with union-find
    fn num_islands(&self) -> u64 {
        fn union(parent: &mut [[Position; SIDE]; SIDE], p: Position, q: Position) {
            let p = find_parent(parent, p);
            let q = find_parent(parent, q);
            if p == q {
                return;
            }
            parent[p.0][p.1] = q
        }

        fn find_parent(parent: &mut [[Position; SIDE]; SIDE], p: Position) -> Position {
            let mut child = p;
            let mut p = p;
            while parent[p.0][p.1] != p {
                p = parent[p.0][p.1];
            }
            // Path compression
            while child != p {
                child = parent[child.0][child.1];
                parent[child.0][child.1] = p;
            }
            p
        }

        let mut parent = [[(usize::MAX, usize::MAX); SIDE]; SIDE];
        POSITIONS.iter().for_each(|&(i, j)| {
            if self.contains(i, j) {
                parent[i][j] = (i, j);
                if i > 0 && self[i - 1][j] == self[i][j] {
                    union(&mut parent, (i, j), (i - 1, j));
                }
                if j > 0 && self[i][j - 1] == self[i][j] {
                    union(&mut parent, (i, j), (i, j - 1));
                }
            }
        });
        POSITIONS
            .iter()
            .fold(0, |count, &(x, y)| count + (parent[x][y] == (x, y)) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl From<[[u64; SIDE]; SIDE]> for Grid {
        fn from(values: [[u64; SIDE]; SIDE]) -> Self {
            let mut grid: [[Option<Tile>; SIDE]; SIDE] = Default::default();
            POSITIONS.iter().for_each(|pos| {
                let value = values[pos.0][pos.1];
                grid[pos.0][pos.1] = if value > 0 {
                    Some(Tile {
                        position: (pos.0, pos.1),
                        value,
                        ..Default::default()
                    })
                } else {
                    None
                }
            });
            Grid::new(grid)
        }
    }

    #[test]
    fn test_find_next_position() {
        let values = [[16, 0, 2, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4, 0]];
        for ((x, y), dir, answer) in vec![
            ((0, 2), Direction::Up, None),
            ((0, 2), Direction::Right, Some((0, 3))),
            ((0, 2), Direction::Left, Some((0, 0))),
            ((0, 2), Direction::Down, Some((3, 2))),
            ((3, 3), Direction::Left, Some((3, 2))),
        ] {
            let grid: Grid = values.into();
            assert_eq!(grid.find_next_position(x, y, &dir), answer);
        }
    }

    #[test]
    fn test_smoothness() {
        for (values, smoothness) in vec![
            (
                [[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 128]],
                0,
            ),
            (
                [
                    [1024, 1024, 1024, 0],
                    [1024, 1024, 1024, 1024],
                    [1024, 1024, 1024, 1024],
                    [1024, 1024, 1024, 1024],
                ],
                0,
            ),
            (
                [[4, 4, 16, 8], [8, 16, 0, 2], [32, 0, 0, 0], [16, 64, 0, 0]],
                -19,
            ),
            (
                [[4, 4, 0, 8], [8, 16, 0, 2], [0, 0, 32, 0], [16, 64, 0, 128]],
                -22,
            ),
            (
                [[4, 4, 16, 8], [8, 16, 0, 2], [32, 0, 32, 0], [16, 64, 0, 2]],
                -25,
            ),
        ] {
            let grid: Grid = values.into();
            assert_eq!(grid.smoothness(), smoothness);
        }
    }

    #[test]
    fn test_monotonicity() {
        for (values, monotonicity) in vec![
            ([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 0),
            (
                [
                    [8, 32, 64, 512],
                    [4, 8, 16, 256],
                    [2, 4, 8, 32],
                    [0, 0, 4, 8],
                ],
                0,
            ),
            (
                [[4, 4, 0, 8], [8, 16, 0, 2], [0, 0, 32, 0], [16, 64, 0, 128]],
                -15,
            ),
            (
                [[4, 4, 16, 8], [8, 16, 0, 2], [32, 0, 32, 0], [16, 64, 0, 2]],
                -13,
            ),
            (
                [
                    [4, 8, 256, 16],
                    [1024, 32, 512, 64],
                    [8, 0, 4, 8],
                    [512, 2, 8, 32],
                ],
                -36,
            ),
        ] {
            let grid: Grid = values.into();
            assert_eq!(grid.monotonicity(), monotonicity);
        }
    }

    #[test]
    fn test_num_islands() {
        for (values, num_islands) in vec![
            ([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 0),
            ([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 8]], 4),
            (
                [[4, 4, 16, 8], [8, 16, 0, 2], [32, 0, 0, 0], [16, 64, 0, 0]],
                9,
            ),
            (
                [[0, 2, 2, 4], [2, 2, 8, 16], [2, 4, 0, 0], [0, 0, 1024, 512]],
                7,
            ),
            (
                [
                    [4, 4, 4, 4],
                    [8, 8, 8, 4],
                    [16, 16, 8, 16],
                    [16, 32, 32, 32],
                ],
                5,
            ),
            (
                [[4, 4, 16, 8], [8, 16, 0, 2], [32, 0, 32, 0], [16, 64, 0, 2]],
                11,
            ),
        ] {
            let grid: Grid = values.into();
            assert_eq!(grid.num_islands(), num_islands);
        }
    }

    #[test]
    fn test_log_2() {
        assert_eq!(log_2(8), 3);
        assert_eq!(log_2(16), 4);
        assert_eq!(log_2(128), 7);
        assert_eq!(log_2(2048), 11);
        assert_eq!(log_2(16_777_216), 24);
    }
}
