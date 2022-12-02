// unused_unit false positives from wasm-bindgen (as of Rust 1.59.0)
#![allow(clippy::unused_unit)]
#![warn(rust_2018_idioms)]

mod minimax;
mod state;

use std::cell::RefCell;
use std::convert::TryFrom;
use std::rc::Rc;
use wasm_bindgen::convert::FromWasmAbi;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::minimax::MinimaxPlayer;
use crate::state::{Direction, Position, State, Tile, SIDE};

type JsResult<T> = Result<T, JsValue>;
type JsError = Result<(), JsValue>;

const AUTO_SAVE: &str = "2048autoSave";
const SNAPSHOT: &str = "2048snapshot";

enum Key {
    Direction(Direction),
    SaveSnapshot,
    RestoreSnapshot,
    ToggleMachinePlay,
}

impl TryFrom<u32> for Key {
    type Error = JsValue;

    fn try_from(code: u32) -> Result<Self, Self::Error> {
        match code {
            83 => Ok(Key::SaveSnapshot),                     // S
            87 => Ok(Key::RestoreSnapshot),                  // W
            80 => Ok(Key::ToggleMachinePlay),                // P
            38 | 75 => Ok(Key::Direction(Direction::Up)),    // Up | Vim up
            37 | 72 => Ok(Key::Direction(Direction::Left)),  // Left | Vim left
            40 | 74 => Ok(Key::Direction(Direction::Down)),  // Down | Vim down
            39 | 76 => Ok(Key::Direction(Direction::Right)), // Right | Vim right
            _ => Err(format!("Invalid code: {}", code).into()),
        }
    }
}

enum Message {
    _Won,
    Lost,
}

struct Board {
    doc: web_sys::Document,
    storage: Option<web_sys::Storage>,

    tile_container: web_sys::Element,
    score_container: web_sys::Element,
    best_score_container: web_sys::Element,
    message_container: web_sys::Element,

    state: State,
}

impl Board {
    fn new(doc: web_sys::Document, storage: Option<web_sys::Storage>) -> JsResult<Board> {
        let grid_container = doc
            .query_selector(".grid-container")?
            .ok_or("couldn't find 'grid-container'")?;

        for _ in 0..SIDE {
            let grid_row_div = doc.create_element("div")?;
            grid_row_div.class_list().add_1("grid-row")?;

            for _ in 0..SIDE {
                let grid_cell_div = doc.create_element("div")?;
                grid_cell_div.class_list().add_1("grid-cell")?;
                grid_row_div.append_child(&grid_cell_div)?;
            }
            grid_container.append_child(&grid_row_div)?;
        }

        Ok(Self {
            tile_container: doc
                .query_selector(".tile-container")?
                .ok_or("couldn't find '.tile-container'")?,
            score_container: doc
                .query_selector(".score-container")?
                .ok_or("couldn't find '.score-container'")?,
            best_score_container: doc
                .query_selector(".best-container")?
                .ok_or("couldn't find '.best-container'")?,
            message_container: doc
                .query_selector(".game-message")?
                .ok_or("couldn't find '.game-message'")?,
            state: State::default(),
            doc,
            storage,
        })
    }

    fn start_game(&mut self) -> JsError {
        self.clear_message()?;
        self.clear_storage()?;
        self.clear_container()?;

        self.state.reset();
        self.render_scores()?;

        self.state.add_tiles(2);
        for tile in self.state.tiles() {
            self.render_tile(tile)?;
        }
        self.save_current_state()
    }

    fn make_move(&mut self, dir: &Direction) -> JsError {
        let updated = self.state.update(dir);
        if !updated {
            return Ok(());
        }
        self.state.add_tiles(1);
        self.render()?;

        if self.state.game_terminated() {
            return self.print_message(Message::Lost);
        }
        Ok(())
    }

    fn render(&mut self) -> JsError {
        self.clear_container()?;

        for tile in self.state.tiles() {
            self.render_tile(tile)?;
        }
        self.render_scores()?;
        self.save_current_state()?;

        Ok(())
    }

    fn render_tile(&self, tile: &Tile) -> JsError {
        fn position_class((x, y): &Position) -> String {
            format!("tile-position-{}-{}", y + 1, x + 1)
        }
        fn apply_classes(element: &web_sys::Element, classes: &[String]) -> JsError {
            element.set_attribute("class", &classes.join(" "))
        }

        let wrapper = self.doc.create_element("div")?;
        let inner = self.doc.create_element("div")?;

        let position = tile.prev_pos.unwrap_or(tile.position);
        let mut classes = vec![
            "tile".to_owned(),
            format!("tile-{}", tile.value),
            position_class(&position),
        ];
        if tile.value > 2048 {
            classes.push("tile-super".to_owned());
        }
        apply_classes(&wrapper, &classes)?;

        inner.class_list().add_1("tile-inner")?;
        inner.set_text_content(Some(&tile.value.to_string()));

        if tile.prev_pos.is_some() {
            let wrapper = wrapper.clone();
            let position = tile.position;

            request_animation_frame(move || {
                classes[2] = position_class(&position);
                apply_classes(&wrapper, &classes)
                    .expect("couldn't apply classes to wrapper `div` object");
            });
        } else if let Some(tiles) = &tile.merged_from {
            classes.push("tile-merged".to_owned());
            apply_classes(&wrapper, &classes)?;

            for merged_tile in (*tiles).iter() {
                self.render_tile(merged_tile)?;
            }
        } else if tile.first_gen {
            classes.push("tile-new".to_owned());
            apply_classes(&wrapper, &classes)?;
        }

        wrapper.append_child(&inner)?;
        self.tile_container.append_child(&wrapper)?;
        Ok(())
    }

    fn clear_container(&self) -> JsError {
        while let Some(first_child) = self.tile_container.first_child() {
            self.tile_container.remove_child(&first_child)?;
        }
        Ok(())
    }

    fn render_scores(&mut self) -> JsError {
        self.score_container
            .set_text_content(Some(&self.state.total_score.to_string()));
        if self.state.delta_score > 0 {
            let animation = self.doc.create_element("div")?;
            animation.class_list().add_1("score-addition")?;
            animation.set_text_content(Some(&format!("+{}", self.state.delta_score)));
            self.score_container.append_child(&animation)?;
        }

        self.state.best_score = std::cmp::max(self.state.best_score, self.state.total_score);
        self.best_score_container
            .set_text_content(Some(&self.state.best_score.to_string()));
        Ok(())
    }

    fn print_message(&mut self, message: Message) -> JsError {
        let (class, message) = match message {
            Message::_Won => ("game-won", "You win!"),
            Message::Lost => ("game-over", "Game over!"),
        };
        self.message_container.class_list().add_1(class)?;
        self.message_container
            .get_elements_by_tag_name("p")
            .item(0)
            .ok_or_else(|| JsValue::from_str("Couldn't find <p> tag in .game-container"))?
            .set_text_content(Some(message));
        Ok(())
    }

    fn clear_message(&mut self) -> JsError {
        self.message_container
            .class_list()
            .remove_2("game-won", "game-over")
    }

    fn restore(&mut self, key: &str) -> JsError {
        self.restore_state(key)?;
        self.save_state(AUTO_SAVE)?; // page refresh should start from restored state

        let tiles = self.state.tiles();
        if tiles.count() == 0 || self.state.game_terminated() {
            return self.start_game();
        }
        self.render()
    }

    fn restore_state(&mut self, key: &str) -> JsError {
        if let Some(storage) = &self.storage {
            if let Some(prev_state) = storage.get_item(key)? {
                self.state.deserialize_from(&prev_state);
            }
        }
        Ok(())
    }

    fn restore_last_game(&mut self) -> JsError {
        self.restore(AUTO_SAVE)
    }

    fn restore_snapshot(&mut self) -> JsError {
        self.restore(SNAPSHOT)
    }

    fn save_state(&mut self, key: &str) -> JsError {
        if let Some(storage) = &self.storage {
            storage.set_item(key, &self.state.serialize())?
        }
        Ok(())
    }

    fn save_current_state(&mut self) -> JsError {
        self.save_state(AUTO_SAVE)
    }

    fn save_snapshot(&mut self) -> JsError {
        self.save_state(SNAPSHOT)
    }

    fn clear_storage(&self) -> JsError {
        if let Some(storage) = &self.storage {
            storage.clear()?;
        }
        Ok(())
    }

    fn find_best_move(&mut self) -> Option<Direction> {
        MinimaxPlayer::new(&mut self.state).find_best_move()
    }
}

fn add_event_listener<E, I, T>(element: I, event: &str, handler: T)
where
    E: 'static + FromWasmAbi,
    I: Into<web_sys::EventTarget>,
    T: 'static + FnMut(E),
{
    let closure = Closure::wrap(Box::new(handler) as Box<dyn FnMut(_)>);
    let target: web_sys::EventTarget = element.into();
    target
        .add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())
        .unwrap();
    closure.forget();
}

fn request_animation_frame<T: 'static + FnMut()>(handler: T) {
    let closure = Closure::wrap(Box::new(handler) as Box<dyn FnMut()>);
    let window = web_sys::window().expect("no global `window` exists");
    window
        .request_animation_frame(closure.as_ref().unchecked_ref())
        .unwrap();
    closure.forget();
}

fn set_interval<T: 'static + FnMut()>(f: T, timeout: i32) -> i32 {
    let closure = Closure::wrap(Box::new(f) as Box<dyn FnMut()>);
    let window = web_sys::window().expect("no global `window` exists");
    let id = window
        .set_interval_with_callback_and_timeout_and_arguments_0(
            closure.as_ref().unchecked_ref(),
            timeout,
        )
        .expect("could not set interval");
    closure.forget();
    id
}

fn clear_interval(handle: i32) {
    let window = web_sys::window().expect("no global `window` exists");
    window.clear_interval_with_handle(handle);
}

#[wasm_bindgen(start)]
pub fn main() -> JsError {
    #[cfg(debug_assertions)]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    let window = web_sys::window().expect("no global `window` exists");
    let doc = window.document().expect("should have a document on window");
    let storage = window
        .local_storage()
        .expect("couldn't access local storage");

    let new_game_button = doc
        .query_selector(".restart-button")?
        .ok_or("couldn't find '.restart-button'")?;
    let retry_button = doc
        .query_selector(".retry-button")?
        .ok_or("couldn't find '.retry-button'")?;

    let board = Rc::new(RefCell::new(Board::new(doc, storage)?));
    board
        .borrow_mut()
        .restore_last_game()
        .expect("couldn't restore last game");

    macro_rules! restart_on_event {
        ($button:expr) => {
            let bc = Rc::clone(&board);
            add_event_listener($button, "click", move |_: web_sys::Event| {
                bc.borrow_mut().start_game().expect("couldn't restart game");
            });
        };
    }
    restart_on_event!(new_game_button);
    restart_on_event!(retry_button);

    let mut interval = None;

    add_event_listener(window, "keydown", move |e: web_sys::KeyboardEvent| {
        if let Ok(key) = Key::try_from(e.key_code()) {
            e.prevent_default();

            let bc = Rc::clone(&board);
            match key {
                Key::Direction(dir) => {
                    request_animation_frame(move || {
                        bc.borrow_mut()
                            .make_move(&dir)
                            .expect("couldn't update board");
                    });
                }
                Key::SaveSnapshot => bc
                    .borrow_mut()
                    .save_snapshot()
                    .expect("couldn't save snapshot"),
                Key::RestoreSnapshot => bc
                    .borrow_mut()
                    .restore_snapshot()
                    .expect("couldn't restore snapshot"),
                Key::ToggleMachinePlay => {
                    if let Some(handle) = interval {
                        interval = None;
                        return clear_interval(handle);
                    }
                    interval = Some(set_interval(
                        move || {
                            if let Some(dir) = bc.borrow_mut().find_best_move() {
                                let bc = Rc::clone(&bc);
                                request_animation_frame(move || {
                                    bc.borrow_mut()
                                        .make_move(&dir)
                                        .expect("couldn't update board");
                                });
                            }
                        },
                        200,
                    ));
                }
            }
        }
    });
    Ok(())
}
