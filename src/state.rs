use rand::distributions::{Distribution, Uniform};
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

pub const SIDE: usize = 4;
const TILES_COUNT: usize = SIDE * SIDE;

#[derive(Copy, Clone)]
pub enum Direction {
    Up = 0,
    Left,
    Down,
    Right,
}

pub type Position = (usize, usize);

#[derive(Deserialize, Serialize)]
struct SaveState {
    score: u64,
    best_score: u64,

    values: Vec<u64>,
}

pub const fn positions() -> [Position; TILES_COUNT] {
    let mut pos: [Position; TILES_COUNT] = [(0, 0); TILES_COUNT];
    let mut i = 0;
    while i < SIDE {
        let mut j = 0;
        while j < SIDE {
            pos[i * SIDE + j] = (i, j);
            j += 1;
        }
        i += 1;
    }
    pos
}
pub const POSITIONS: [Position; TILES_COUNT] = positions();

const fn build_traversals(dir: &Direction) -> [[Position; SIDE]; SIDE] {
    let mut v: [[Position; SIDE]; SIDE] = [[(0, 0); SIDE]; SIDE];
    let mut i = 0;
    while i < SIDE {
        let mut vv: [Position; SIDE] = [(0, 0); SIDE];
        let mut j = 0;
        while j < SIDE {
            vv[j] = match dir {
                Direction::Up => (j, i),
                Direction::Down => (SIDE - j - 1, i),
                Direction::Left => (i, j),
                Direction::Right => (i, SIDE - j - 1),
            };
            j += 1;
        }
        v[i] = vv;
        i += 1;
    }
    v
}
pub const TRAVERSALS: [[[Position; SIDE]; SIDE]; 4] = [
    build_traversals(&Direction::Up),
    build_traversals(&Direction::Left),
    build_traversals(&Direction::Down),
    build_traversals(&Direction::Right),
];

// Occupied cell, cannot be empty
#[derive(Default, Clone)]
pub struct Tile {
    pub position: Position,
    pub value: u64,
    pub prev_pos: Option<Position>,
    pub merged_from: Option<Box<[Tile; 2]>>,
    pub first_gen: bool, // created in this turn
}

impl PartialEq for Tile {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Tile {
    pub fn new(x: usize, y: usize, value: u64) -> Self {
        Self {
            position: (x, y),
            value,
            first_gen: true,
            ..Default::default()
        }
    }

    fn random(x: usize, y: usize) -> Self {
        Self {
            position: (x, y),
            value: if Uniform::from(0..10).sample(&mut rand::thread_rng()) != 9 {
                2
            } else {
                4
            },
            first_gen: true,
            ..Default::default()
        }
    }
}

#[derive(Default, Clone)]
pub struct Grid([[Option<Tile>; SIDE]; SIDE]);

pub struct GridUpdate {
    pub success: bool,

    total_score: u64,
    delta_score: u64,
}

impl Index<usize> for Grid {
    type Output = [Option<Tile>; SIDE];

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Grid {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Grid {
    pub fn new(grid: [[Option<Tile>; SIDE]; SIDE]) -> Self {
        Grid(grid)
    }

    pub fn tiles(&self) -> impl Iterator<Item = &Tile> {
        POSITIONS
            .iter()
            .filter_map(move |&(x, y)| self[x][y].as_ref())
    }

    pub fn add_random(&mut self, count: usize) {
        assert!(count <= TILES_COUNT);
        for &(x, y) in POSITIONS
            .iter()
            .filter(|&&(x, y)| !self.contains(x, y))
            .choose_multiple(&mut rand::thread_rng(), count)
        {
            self[x][y] = Some(Tile::random(x, y));
        }
    }

    pub fn add(&mut self, x: usize, y: usize, val: u64) {
        self[x][y] = Some(Tile::new(x, y, val))
    }

    pub fn remove(&mut self, x: usize, y: usize) {
        self[x][y] = None
    }

    pub fn contains(&self, x: usize, y: usize) -> bool {
        self[x][y].is_some()
    }

    pub fn update(&mut self, dir: &Direction) -> GridUpdate {
        self.cleanup();

        let mut success = false;
        let mut total_score = 0;
        let mut delta_score = 0;

        for traversal in TRAVERSALS[*dir as usize].iter() {
            let (mut i, mut j) = (0, 0);
            while i < traversal.len() && j < traversal.len() {
                let (ix, iy) = traversal[i];
                let (jx, jy) = traversal[j];

                if !self.contains(jx, jy) || i == j {
                    j += 1;
                    continue;
                }
                match (&self[ix][iy], &self[jx][jy]) {
                    (Some(tile_i), Some(tile_j)) if tile_i.value == tile_j.value => {
                        let new_val = tile_i.value * 2;
                        self[ix][iy] = Some(Tile {
                            position: (ix, iy),
                            value: new_val,
                            prev_pos: Some((jx, jy)),
                            merged_from: Some(Box::new([
                                Tile {
                                    position: (ix, iy),
                                    value: tile_i.value,
                                    ..Default::default()
                                },
                                Tile {
                                    position: (jx, jy),
                                    value: tile_j.value,
                                    ..Default::default()
                                },
                            ])),
                            ..Default::default()
                        });
                        self.remove(jx, jy);
                        total_score += new_val;
                        delta_score += new_val;
                        success = true;
                        i += 1;
                        j += 1;
                    }
                    (None, Some(tile_j)) => {
                        self[ix][iy] = Some(Tile {
                            position: (ix, iy),
                            value: tile_j.value,
                            prev_pos: Some((jx, jy)),
                            ..Default::default()
                        });
                        self.remove(jx, jy);
                        success = true;
                        j += 1;
                    }
                    _ => {
                        i += 1;
                    }
                };
            }
        }
        GridUpdate {
            success,
            total_score,
            delta_score,
        }
    }

    fn game_terminated(&self) -> bool {
        if self.tiles().count() != TILES_COUNT {
            return false;
        }
        !self.can_merge(&Direction::Right) && !self.can_merge(&Direction::Down)
    }

    // checks whether two neighboring cells can merge in a certain direction
    // makes sense for boards without empty cells
    fn can_merge(&self, dir: &Direction) -> bool {
        for traversal in TRAVERSALS[*dir as usize].iter() {
            let (mut i, mut j) = (1, 0);

            while i < traversal.len() {
                let (ix, iy) = traversal[i];
                let (jx, jy) = traversal[j];

                match (&self[ix][iy], &self[jx][jy]) {
                    (Some(tile_x), Some(tile_y)) if tile_x.value == tile_y.value => return true,
                    _ => {
                        i += 1;
                        j += 1;
                    }
                }
            }
        }
        false
    }

    fn cleanup(&mut self) {
        POSITIONS.iter().for_each(|&(x, y)| {
            if let Some(tile) = self[x][y].as_mut() {
                tile.prev_pos = None;
                tile.merged_from = None;
                tile.first_gen = false;
            }
        });
    }
}

#[derive(Default)]
pub struct State {
    pub total_score: u64,
    pub best_score: u64,
    pub delta_score: u64, // new score difference

    pub grid: Grid,
}

impl State {
    pub fn tiles(&self) -> impl Iterator<Item = &Tile> {
        self.grid.tiles()
    }

    pub fn add_tiles(&mut self, num: usize) {
        self.grid.add_random(num)
    }

    pub fn update(&mut self, dir: &Direction) -> bool {
        self.delta_score = 0;
        let GridUpdate {
            success,
            total_score,
            delta_score,
        } = self.grid.update(dir);
        self.total_score += total_score;
        self.delta_score += delta_score;
        success
    }

    pub fn reset(&mut self) {
        let _ = std::mem::replace(
            self,
            Self {
                best_score: self.best_score,
                ..Default::default()
            },
        );
    }

    pub fn game_terminated(&self) -> bool {
        self.grid.game_terminated()
    }

    pub fn serialize(&self) -> String {
        let values: Vec<_> = POSITIONS
            .iter()
            .map(|&(x, y)| self.grid[x][y].as_ref().map(|tile| tile.value).unwrap_or(0))
            .collect();
        serde_json::to_string(&SaveState {
            score: self.total_score,
            best_score: self.best_score,
            values,
        })
        .expect("error serializing to json")
    }

    pub fn deserialize_from(&mut self, prev_state: &str) {
        let state: SaveState = serde_json::from_str(prev_state).expect("error deserializing state");

        let mut grid: [[Option<Tile>; SIDE]; SIDE] = Default::default();
        for (&value, &(x, y)) in state.values.iter().zip(POSITIONS.iter()) {
            grid[x][y] = if value > 0 {
                Some(Tile {
                    position: (x, y),
                    value,
                    first_gen: true,
                    ..Default::default()
                })
            } else {
                None
            };
        }
        let _ = std::mem::replace(
            self,
            Self {
                total_score: state.score,
                best_score: state.best_score,
                grid: Grid::new(grid),
                ..Default::default()
            },
        );
    }
}
