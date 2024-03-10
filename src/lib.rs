#![deny(unused_qualifications)]
#![deny(clippy::semicolon_if_nothing_returned)]

use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use ahash::RandomState;
use dashmap::{DashMap, RwLockWriteGuard, SharedValue};
use hashbrown::hash_map::RawEntryMut;
use hashbrown::HashMap;
use triomphe::Arc;

type InternSet<T> = DashMap<Arc<T>, (), RandomState>;
type Guard<T> = RwLockWriteGuard<'static, HashMap<Arc<T>, SharedValue<()>, RandomState>>;

#[allow(clippy::declare_interior_mutable_const)]
const NOOP: SharedValue<()> = SharedValue::new(());

pub struct InternStorage<T: ?Sized> {
    map: std::sync::OnceLock<InternSet<T>>,
}

impl<T: Internable + ?Sized> InternStorage<T> {
    fn get(&self) -> &InternSet<T> {
        self.map.get_or_init(<_>::default)
    }
}

impl<T: ?Sized> InternStorage<T> {
    pub const fn new() -> Self {
        Self { map: std::sync::OnceLock::new() }
    }
}

impl<T: ?Sized> Default for InternStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub trait Internable: Hash + Eq + 'static {
    fn storage() -> &'static InternStorage<Self>;

    fn intern(self) -> Interned<Self>
    where
        Self: Sized,
    {
        Interned::new(self)
    }
}

pub struct Interned<T: Internable + ?Sized>(Arc<T>);

impl<T: Internable> Interned<T> {
    /// Intern a value.
    pub fn new(value: T) -> Self {
        let (mut shard, hash) = Self::select(&value);

        match shard.raw_entry_mut().from_key_hashed_nocheck(hash, &value) {
            RawEntryMut::Occupied(occ) => Self(occ.key().clone()),
            RawEntryMut::Vacant(vac) => {
                Self(vac.insert_hashed_nocheck(hash, Arc::new(value), NOOP).0.clone())
            }
        }
    }
}

impl Interned<str> {
    /// Interns the given string.
    ///
    /// # Example
    ///
    /// ```
    /// use amuro::Interned;
    ///
    /// let a = Interned::new_str("amuro");
    /// let b = Interned::new_str("amuro");
    ///
    /// assert_eq!(a, b);
    /// ```
    pub fn new_str(s: &str) -> Self {
        let (mut shard, hash) = Self::select(s);

        match shard.raw_entry_mut().from_key_hashed_nocheck(hash, s) {
            RawEntryMut::Occupied(occupied) => Self(occupied.key().clone()),
            RawEntryMut::Vacant(vacant) => {
                Self(vacant.insert_hashed_nocheck(hash, Arc::from(s), NOOP).0.clone())
            }
        }
    }
}

impl<T: Internable + ?Sized> Interned<T> {
    fn select(value: &T) -> (Guard<T>, u64) {
        let storage = T::storage().get();
        let hash = storage.hasher().hash_one(value);
        let shard_idx = storage.determine_shard(hash as usize);
        let shard = storage.shards()[shard_idx].write();

        (shard, hash)
    }
}

impl<T: Internable + ?Sized> Clone for Interned<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Internable + ?Sized> Deref for Interned<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Internable + ?Sized> PartialEq for Interned<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T: Internable + ?Sized> Eq for Interned<T> {}

impl<T: Internable + PartialOrd + ?Sized> PartialOrd for Interned<T> {
    fn partial_cmp(&self, other: &Interned<T>) -> Option<Ordering> {
        if Arc::ptr_eq(&self.0, &other.0) {
            Some(Ordering::Equal)
        } else {
            self.0.partial_cmp(&other.0)
        }
    }
}

impl<T: Internable + Ord + ?Sized> Ord for Interned<T> {
    fn cmp(&self, other: &Interned<T>) -> Ordering {
        if Arc::ptr_eq(&self.0, &other.0) { Ordering::Equal } else { self.0.cmp(&other.0) }
    }
}

impl<T: Internable + ?Sized> Hash for Interned<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(Arc::as_ptr(&self.0) as *const () as usize);
    }
}

impl<T: Debug + Internable + ?Sized> Debug for Interned<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (*self.0).fmt(f)
    }
}

impl<T: Display + Internable + ?Sized> Display for Interned<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (*self.0).fmt(f)
    }
}

impl<T: Internable + ?Sized> Drop for Interned<T> {
    fn drop(&mut self) {
        if Arc::count(&self.0) == 2 {
            let (mut shard, hash) = Self::select(&self.0);

            match shard.raw_entry_mut().from_key_hashed_nocheck(hash, &self.0) {
                RawEntryMut::Occupied(occ) => occ.remove(),
                RawEntryMut::Vacant(_) => unreachable(),
            };

            if shard.len() * 2 < shard.capacity() {
                shard.shrink_to_fit();
            }
        }
    }
}

#[macro_export]
macro_rules! impl_internable {
    ( $($t:path),+ $(,)? ) => { $(
        impl $crate::Internable for $t {
            fn storage() -> &'static $crate::InternStorage<Self> {
                static STORAGE: $crate::InternStorage<$t> = $crate::InternStorage::new();
                &STORAGE
            }
        }
    )+ };
}

impl_internable!(str);

/// Next best thing after `core::hint::unreachable_unchecked()`
/// If happens to be called this will stall CPU, instead of causing UB.
#[inline]
#[cold]
#[allow(clippy::empty_loop)]
const fn unreachable() -> ! {
    loop {}
}

#[cfg(test)]
mod tests {
    use hashbrown::HashSet;

    use crate::{Internable, Interned};

    #[test]
    fn smoke() {
        let a = Interned::new_str("amuro");
        let b = Interned::new_str("amuro");

        assert_eq!(a, b);

        let storage = str::storage().get();
        assert_eq!(storage.len(), 1);

        let mut set = HashSet::new();
        for value in ["amuro"].repeat(10).into_iter().map(ToString::to_string) {
            set.insert(Interned::new_str(value.as_str()));
        }
        assert_eq!(set.len(), 1);
    }
}
