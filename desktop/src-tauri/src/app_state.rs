use std::{
    collections::HashMap,
    io::Write,
    sync::{
        atomic::{AtomicBool, AtomicU16},
        Arc, Mutex,
    },
};

use nostr::{Keys, ToBech32};
use tauri::{AppHandle, Manager};
#[cfg(feature = "mesh-llm")]
use tokio::sync::Mutex as AsyncMutex;

use crate::huddle::HuddleState;
use crate::managed_agents::config_bridge::SessionConfigCache;
use crate::managed_agents::ManagedAgentProcess;
pub struct AppState {
    pub keys: Mutex<Keys>,
    pub http_client: reqwest::Client,
    /// Workspace-provided relay URL override. Set by `apply_workspace` on app
    /// init and takes priority over env vars and compile-time defaults.
    pub relay_url_override: Mutex<Option<String>>,
    pub managed_agents_store_lock: Mutex<()>,
    pub channel_templates_store_lock: Mutex<()>,
    pub managed_agent_processes: Mutex<HashMap<String, ManagedAgentProcess>>,
    pub huddle_state: Mutex<HuddleState>,
    /// Tauri app handle — stored after setup so huddle commands can emit
    /// `huddle-state-changed` events without needing the handle threaded
    /// through every call site.
    ///
    /// Set once during `setup()` in `lib.rs`; never cleared.
    pub app_handle: Mutex<Option<AppHandle>>,
    /// Selected audio output device name. `None` = system default.
    /// Used by `connect_audio_relay` and TTS pipeline when opening sinks.
    pub audio_output_device: Mutex<Option<String>>,
    /// Port of the localhost media streaming proxy (set during setup).
    pub media_proxy_port: AtomicU16,
    /// Set when identity resolution detected a "lost" state: the migration
    /// marker was present but the keyring was empty and no plaintext fallback
    /// existed. An ephemeral key was generated to let the app boot; the
    /// frontend checks this flag via `get_identity` and routes to the nsec
    /// re-import step instead of the normal onboarding profile flow.
    pub identity_lost: AtomicBool,
    /// Cached ACP session config from running agents, keyed by agent pubkey.
    /// Populated when the harness emits `session_config_captured` observer events.
    pub session_config_cache: Mutex<HashMap<String, SessionConfigCache>>,
    /// IOKit power assertion state — prevents idle sleep while agents run.
    pub prevent_sleep: Arc<Mutex<crate::prevent_sleep::PreventSleepState>>,
    /// In-process mesh-llm node started by Buzz Desktop.
    #[cfg(feature = "mesh-llm")]
    pub mesh_llm_runtime: AsyncMutex<Option<crate::mesh_llm::DesktopMeshRuntime>>,
    /// Runtime-owned relay-mesh control plane (call-me-now listener + connect
    /// request publish/retry). Installed once at identity-set time so the
    /// listener is up before any restore/create can request a connection.
    #[cfg(feature = "mesh-llm")]
    pub mesh_coordinator: AsyncMutex<Option<crate::mesh_llm::MeshCoordinator>>,
}

/// Parse the `BUZZ_PRIVATE_KEY` env var into identity keys. `Some` means the
/// env var was present and valid and MUST win over any persisted/keyring key
/// (the dev/CI/harness override). `None` means absent or malformed — callers
/// fall through to persisted resolution. A malformed value is logged and
/// treated as absent rather than left on an ephemeral identity.
fn identity_from_env() -> Option<Keys> {
    match std::env::var("BUZZ_PRIVATE_KEY") {
        Ok(nsec) => match Keys::parse(nsec.trim()) {
            Ok(keys) => Some(keys),
            Err(error) => {
                eprintln!("buzz-desktop: invalid BUZZ_PRIVATE_KEY: {error}");
                None
            }
        },
        Err(std::env::VarError::NotUnicode(_)) => {
            eprintln!("buzz-desktop: BUZZ_PRIVATE_KEY contains invalid UTF-8");
            None
        }
        Err(std::env::VarError::NotPresent) => None,
    }
}

pub fn build_app_state() -> AppState {
    // Env var takes precedence (dev/CI). If absent, resolve_persisted_identity()
    // in setup() will replace the ephemeral placeholder with a persisted key.
    let keys = match identity_from_env() {
        Some(keys) => {
            eprintln!(
                "buzz-desktop: configured identity pubkey {}",
                keys.public_key().to_hex()
            );
            keys
        }
        None => Keys::generate(),
    };

    AppState {
        keys: Mutex::new(keys),
        http_client: reqwest::Client::builder()
            .resolve("localhost", std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
            .pool_idle_timeout(std::time::Duration::from_secs(10))
            .pool_max_idle_per_host(1)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new()),
        relay_url_override: Mutex::new(None),
        managed_agents_store_lock: Mutex::new(()),
        channel_templates_store_lock: Mutex::new(()),
        managed_agent_processes: Mutex::new(HashMap::new()),
        session_config_cache: Mutex::new(HashMap::new()),
        huddle_state: Mutex::new(HuddleState::default()),
        app_handle: Mutex::new(None),
        audio_output_device: Mutex::new(None),
        media_proxy_port: AtomicU16::new(0),
        prevent_sleep: Arc::new(Mutex::new(
            crate::prevent_sleep::PreventSleepState::default(),
        )),
        identity_lost: AtomicBool::new(false),
        #[cfg(feature = "mesh-llm")]
        mesh_llm_runtime: AsyncMutex::new(None),
        #[cfg(feature = "mesh-llm")]
        mesh_coordinator: AsyncMutex::new(None),
    }
}

impl AppState {
    /// Lock the huddle state mutex, converting a poisoned-lock error to a String.
    ///
    /// Convenience wrapper — replaces 15+ instances of
    /// `state.huddle_state.lock().map_err(|e| e.to_string())?` throughout the
    /// huddle module.
    pub fn huddle(&self) -> Result<std::sync::MutexGuard<'_, crate::huddle::HuddleState>, String> {
        self.huddle_state.lock().map_err(|e| e.to_string())
    }

    pub fn get_session_cache(&self, pubkey: &str) -> Option<SessionConfigCache> {
        self.session_config_cache.lock().ok()?.get(pubkey).cloned()
    }

    pub fn put_session_cache(&self, pubkey: &str, cache: SessionConfigCache) {
        if let Ok(mut map) = self.session_config_cache.lock() {
            map.insert(pubkey.to_string(), cache);
        }
    }

    pub fn clear_session_cache(&self, pubkey: &str) {
        if let Ok(mut map) = self.session_config_cache.lock() {
            map.remove(pubkey);
        }
    }

    /// Emit the current huddle state to the frontend via Tauri event.
    ///
    /// Acquires both locks (app_handle + huddle_state), clones a snapshot,
    /// releases both, then emits. Best-effort — no-op if either lock is
    /// poisoned or the app_handle hasn't been set yet.
    pub fn emit_huddle_state_changed(&self) {
        let app = match self.app_handle.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => return,
        };
        let Some(app) = app else { return };
        let snapshot = match self.huddle_state.lock() {
            Ok(hs) => hs.clone(),
            Err(_) => return,
        };
        crate::huddle::state::emit_huddle_state(&app, &snapshot);
    }
}

/// Resolve the user's identity key from the app data directory.
///
/// Priority: `BUZZ_PRIVATE_KEY` env var (already handled in `build_app_state`)
/// → `{app_data_dir}/identity.key` file → generate + save.
///
/// Writes use `atomic-write-file` which handles temp file creation, fsync,
/// atomic rename, and directory sync — no partial or corrupt files on disk.
///
/// Sets `state.identity_lost` when the keyring held a migration marker but
/// was empty with no plaintext fallback — the app boots with an ephemeral key
/// and the frontend is expected to prompt the user to re-import their nsec.
pub fn resolve_persisted_identity(app: &AppHandle, state: &AppState) -> Result<(), String> {
    // Only skip file-based resolution if the env var was present AND parsed
    // successfully. A malformed env var should fall through to the persisted
    // key rather than leaving the app on an ephemeral identity.
    if identity_from_env().is_some() {
        return Ok(());
    }

    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("app data dir: {e}"))?;
    std::fs::create_dir_all(&data_dir).map_err(|e| format!("create app data dir: {e}"))?;

    let resolved = load_or_create_identity(&data_dir)?;
    state
        .identity_lost
        .store(resolved.lost, std::sync::atomic::Ordering::Relaxed);
    *state.keys.lock().map_err(|e| e.to_string())? = resolved.keys;
    Ok(())
}

/// Service name for the desktop OS keyring. Shared by the human identity key
/// and managed-agent keys (each addressed by a distinct key name within it).
pub(crate) const KEYRING_SERVICE: &str = "buzz-desktop";

/// Keyring key name for the human identity nsec.
pub(crate) const IDENTITY_KEY_NAME: &str = "identity";

/// Filename of the marker written once a successful keyring migration deletes
/// the legacy `identity.key`. Its presence is the only durable signal that a
/// key once lived in the keyring — used to tell a genuine first-ever launch
/// (no key anywhere, generating is correct) from a post-migration boot whose
/// keyring is merely unreachable (the key IS in the keyring, must NOT generate).
const MIGRATION_MARKER_NAME: &str = "identity.migrated";

/// The output of identity resolution. `lost = true` means the keyring was
/// reachable-but-empty after a prior successful migration (marker present, no
/// file) — the key vanished from the keyring externally. An ephemeral key is
/// provided so the app can boot; the frontend must prompt re-import.
struct ResolvedIdentity {
    keys: Keys,
    lost: bool,
}

/// The keyring operations the identity resolution flow needs. Abstracted so the
/// corrupt-keyring recovery decision ([`recover_from_keyring`]) can be
/// unit-tested against a fake without touching the live OS keyring.
trait IdentityKeyStore {
    fn probe(&self, name: &str) -> crate::secret_store::KeyringProbe;
    fn load(&self, name: &str) -> Result<Option<String>, String>;
    fn store(&self, name: &str, value: &str) -> Result<(), String>;
    fn delete(&self, name: &str) -> Result<(), String>;
}

impl IdentityKeyStore for crate::secret_store::SecretStore {
    fn probe(&self, name: &str) -> crate::secret_store::KeyringProbe {
        crate::secret_store::SecretStore::probe(self, name)
    }
    fn load(&self, name: &str) -> Result<Option<String>, String> {
        crate::secret_store::SecretStore::load(self, name)
    }
    fn store(&self, name: &str, value: &str) -> Result<(), String> {
        crate::secret_store::SecretStore::store(self, name, value)
    }
    fn delete(&self, name: &str) -> Result<(), String> {
        crate::secret_store::SecretStore::delete(self, name)
    }
}

/// Resolve the human identity key: migrate a legacy `identity.key` into the
/// keyring when safe, otherwise load from whichever backend holds it, else
/// generate-and-save.
///
/// Migration rule (prevents stale-key resurrection): only import the plaintext
/// file when the keyring is REACHABLE-but-empty. If the keyring is UNREACHABLE
/// this boot, fall back to reading the file directly and do NOT migrate — a
/// later import from a leftover (possibly rotated) file could resurrect an old
/// key.
fn load_or_create_identity(data_dir: &std::path::Path) -> Result<ResolvedIdentity, String> {
    let legacy_path = data_dir.join("identity.key");

    // No keyring available in this build: the `0o600` file is the only store.
    if !cfg!(feature = "system-keyring") {
        let keys = load_file_or_generate(&legacy_path, data_dir)?;
        return Ok(ResolvedIdentity { keys, lost: false });
    }

    let store = crate::secret_store::SecretStore::shared(KEYRING_SERVICE);
    resolve_identity_with_store(store, &legacy_path, data_dir)
}

/// Identity resolution over an [`IdentityKeyStore`] seam. Split from
/// [`load_or_create_identity`] so the probe/recover branches are testable
/// without the live OS keyring.
fn resolve_identity_with_store(
    store: &impl IdentityKeyStore,
    legacy_path: &std::path::Path,
    data_dir: &std::path::Path,
) -> Result<ResolvedIdentity, String> {
    use crate::secret_store::KeyringProbe;

    match store.probe(IDENTITY_KEY_NAME) {
        KeyringProbe::Present => {
            if let Some(nsec) = store.load(IDENTITY_KEY_NAME)? {
                match Keys::parse(nsec.trim()) {
                    Ok(keyring_keys) => {
                        eprintln!(
                            "buzz-desktop: persisted identity pubkey {}",
                            keyring_keys.public_key().to_hex()
                        );
                        // Check for a leftover identity.key. If it holds a
                        // DIFFERENT pubkey, the user imported that key after
                        // the last boot (pre-fix, import only wrote the file).
                        // Adopt it into the keyring so the user's intent sticks.
                        // If the pubkeys match it is a stale leftover from a
                        // prior migration whose remove_file failed — clean it up.
                        if legacy_path.exists() {
                            match load_key_file(legacy_path) {
                                Ok(file_keys)
                                    if file_keys.public_key() != keyring_keys.public_key() =>
                                {
                                    let file_nsec = file_keys
                                        .secret_key()
                                        .to_bech32()
                                        .map_err(|e| format!("encode nsec: {e}"))?;
                                    eprintln!(
                                        "buzz-desktop: identity.key differs from keyring; \
                                         adopting imported key {}",
                                        file_keys.public_key().to_hex()
                                    );
                                    store.store(IDENTITY_KEY_NAME, &file_nsec)?;
                                    match store.load(IDENTITY_KEY_NAME)? {
                                        Some(stored) if stored == file_nsec => {}
                                        _ => {
                                            return Err("keyring read-back verify failed \
                                                 after adopting identity.key"
                                                .to_string());
                                        }
                                    }
                                    if let Err(e) = std::fs::remove_file(legacy_path) {
                                        eprintln!(
                                            "buzz-desktop: failed to delete identity.key \
                                             after adopting: {e}"
                                        );
                                    }
                                    return Ok(ResolvedIdentity {
                                        keys: file_keys,
                                        lost: false,
                                    });
                                }
                                // Same pubkey (stale leftover) or corrupt file
                                // — keyring is authoritative; clean up plaintext.
                                _ => cleanup_leftover_identity_file(legacy_path),
                            }
                        }
                        return Ok(ResolvedIdentity {
                            keys: keyring_keys,
                            lost: false,
                        });
                    }
                    // The corruption is in the KEYRING, not the file. Clear the
                    // bad keyring value and recover from the file (or generate
                    // fresh) — do NOT quarantine a valid leftover `identity.key`
                    // that holds the user's only good key.
                    Err(error) => {
                        let keys =
                            recover_from_keyring(store, legacy_path, data_dir, &error.to_string())?;
                        return Ok(ResolvedIdentity { keys, lost: false });
                    }
                }
            }
            // Probe said Present but load found nothing — treat as empty.
        }
        KeyringProbe::ReachableButEmpty => {
            // One-time migration: import the legacy plaintext file, read-back
            // verify, THEN delete it.
            if legacy_path.exists() {
                if let Some(keys) = migrate_identity_file(store, legacy_path, data_dir)? {
                    return Ok(ResolvedIdentity { keys, lost: false });
                }
            } else if migration_marker_path(data_dir).exists() {
                // Marker present, keyring empty, no file — the key was previously
                // durably stored in the keyring but is now gone (keyring cleared,
                // new login session, or the entry was externally deleted). There
                // is no plaintext fallback to recover from.
                //
                // Generate an ephemeral in-memory key so the app can boot, but
                // surface a "lost" flag so the frontend prompts re-import rather
                // than silently starting a fresh identity.
                let ephemeral = Keys::generate();
                eprintln!(
                    "buzz-desktop: identity lost — keyring was empty despite migration marker; \
                     using ephemeral key {}, awaiting user re-import",
                    ephemeral.public_key().to_hex()
                );
                return Ok(ResolvedIdentity {
                    keys: ephemeral,
                    lost: true,
                });
            }
        }
        KeyringProbe::Unreachable => {
            // Keyring down this boot. If a recoverable file is present, use it
            // (and do NOT migrate — re-importing later could resurrect a
            // rotated key). With NO file, the marker disambiguates two states
            // that are otherwise byte-identical (Unreachable + no file):
            //   - marker present → the key was migrated into the keyring and the
            //     file deleted. The real key is unreachable, not gone. Fail
            //     CLOSED — generating here would silently rotate the identity.
            //   - no marker → genuine first-ever launch with nothing to protect.
            //     Generate to the `0o600` file (legitimate first-run).
            if !legacy_path.exists() && migration_marker_path(data_dir).exists() {
                return Err(
                    "identity key is in the OS keyring but the keyring is unavailable this boot; \
                     retry once the keyring (Keychain / Credential Manager / Secret Service) is reachable"
                        .to_string(),
                );
            }
            let keys = load_file_or_generate(legacy_path, data_dir)?;
            return Ok(ResolvedIdentity { keys, lost: false });
        }
    }

    let keys = generate_and_persist(store, legacy_path, data_dir)?;
    Ok(ResolvedIdentity { keys, lost: false })
}

/// Recover from a corrupt nsec in the keyring (parse failed). Clear the bad
/// keyring value, then migrate a valid leftover `identity.key` if one exists,
/// generating fresh only as a last resort. The keyring delete is best-effort:
/// a delete failure logs and continues — it must never block startup.
fn recover_from_keyring(
    store: &impl IdentityKeyStore,
    legacy_path: &std::path::Path,
    data_dir: &std::path::Path,
    error: &str,
) -> Result<Keys, String> {
    eprintln!("buzz-desktop: corrupt nsec in keyring ({error}), clearing and recovering from file");
    if let Err(e) = store.delete(IDENTITY_KEY_NAME) {
        eprintln!("buzz-desktop: failed to clear corrupt keyring value: {e}");
    }
    if legacy_path.exists() {
        if let Some(keys) = migrate_identity_file(store, legacy_path, data_dir)? {
            return Ok(keys);
        }
    }
    generate_and_persist(store, legacy_path, data_dir)
}

/// Load the `0o600` identity file, quarantining corruption, else generate and
/// save a fresh key to the file. Used when no keyring is available.
fn load_file_or_generate(
    legacy_path: &std::path::Path,
    data_dir: &std::path::Path,
) -> Result<Keys, String> {
    if legacy_path.exists() {
        match load_key_file(legacy_path) {
            Ok(keys) => {
                eprintln!(
                    "buzz-desktop: persisted identity pubkey {}",
                    keys.public_key().to_hex()
                );
                return Ok(keys);
            }
            Err(error) => quarantine_corrupt_key(legacy_path, data_dir, &error),
        }
    }
    let keys = Keys::generate();
    save_key_file(legacy_path, &keys)?;
    eprintln!(
        "buzz-desktop: generated and saved identity pubkey {}",
        keys.public_key().to_hex()
    );
    Ok(keys)
}

/// Import the plaintext `identity.key` into the store, verify the round-trip,
/// then delete the file. Returns `Ok(None)` if the file was corrupt (caller
/// continues to generate-and-save).
fn migrate_identity_file(
    store: &impl IdentityKeyStore,
    legacy_path: &std::path::Path,
    data_dir: &std::path::Path,
) -> Result<Option<Keys>, String> {
    let keys = match load_key_file(legacy_path) {
        Ok(keys) => keys,
        Err(error) => {
            eprintln!("buzz-desktop: corrupt identity.key during migration ({error}), skipping");
            return Ok(None);
        }
    };
    let nsec = keys
        .secret_key()
        .to_bech32()
        .map_err(|e| format!("encode nsec: {e}"))?;

    store.store(IDENTITY_KEY_NAME, &nsec)?;
    // Read-back verify before deleting the plaintext file.
    match store.load(IDENTITY_KEY_NAME)? {
        Some(stored) if stored == nsec => {
            // Crash-safe ordering: record that the key now lives in the keyring
            // (marker write + fsync) BEFORE deleting the file. A crash between
            // the two must never leave "file gone, no marker" — that state is
            // indistinguishable from a fresh install and would silently rotate
            // the identity on the next keyring-unreachable boot. If the marker
            // cannot be written, keep the file so the key is never stranded.
            let marker_path = migration_marker_path(data_dir);
            if let Err(e) = write_migration_marker(&marker_path) {
                eprintln!(
                    "buzz-desktop: keyring import ok but failed to write migration marker ({e}); \
                     keeping identity.key so the key is not stranded"
                );
                return Ok(Some(keys));
            }
            if let Err(e) = std::fs::remove_file(legacy_path) {
                eprintln!("buzz-desktop: keyring import ok but failed to delete identity.key: {e}");
            } else {
                eprintln!("buzz-desktop: migrated identity key into OS keyring");
            }
            Ok(Some(keys))
        }
        _ => Err("keyring read-back verify failed for identity key".to_string()),
    }
}

/// Persist `keys` into the keyring with read-back verification, write the
/// migration marker, and delete any leftover `identity.key`. Returns `Ok` on
/// success. Returns `Err` when the keyring write fails (availability error) —
/// the caller must fall back to `save_key_file` so the key survives the boot.
///
/// This is the shared kernel used by both one-time file migration and the
/// `import_identity` command. Crash-safe ordering: marker is written BEFORE
/// deleting the file.
fn persist_identity_to_keyring(
    store: &impl IdentityKeyStore,
    keys: &Keys,
    legacy_path: &std::path::Path,
    data_dir: &std::path::Path,
) -> Result<(), String> {
    let nsec = keys
        .secret_key()
        .to_bech32()
        .map_err(|e| format!("encode nsec: {e}"))?;

    // Will error if the keyring is unavailable — caller falls back to the file.
    store.store(IDENTITY_KEY_NAME, &nsec)?;

    // Read-back verify before touching durable state.
    match store.load(IDENTITY_KEY_NAME)? {
        Some(stored) if stored == nsec => {}
        _ => return Err("keyring read-back verify failed".to_string()),
    }

    // Write marker before deleting the file (crash-safe ordering).
    let marker_path = migration_marker_path(data_dir);
    if let Err(e) = write_migration_marker(&marker_path) {
        eprintln!(
            "buzz-desktop: keyring write ok but failed to write migration marker ({e}); \
             keeping identity.key so the key is not stranded"
        );
        return Ok(());
    }

    if legacy_path.exists() {
        if let Err(e) = std::fs::remove_file(legacy_path) {
            eprintln!("buzz-desktop: keyring write ok but failed to delete identity.key: {e}");
        }
    }

    Ok(())
}

/// Public-crate wrapper around [`persist_identity_to_keyring`] for use by the
/// `import_identity` Tauri command. Takes the concrete [`SecretStore`] type so
/// the command does not need visibility into the private `IdentityKeyStore`
/// trait.
pub(crate) fn import_identity_to_keyring(
    store: &crate::secret_store::SecretStore,
    keys: &Keys,
    legacy_path: &std::path::Path,
    data_dir: &std::path::Path,
) -> Result<(), String> {
    persist_identity_to_keyring(store, keys, legacy_path, data_dir)
}

/// Path of the migration-completed marker within `data_dir`.
fn migration_marker_path(data_dir: &std::path::Path) -> std::path::PathBuf {
    data_dir.join(MIGRATION_MARKER_NAME)
}

/// Atomically write (and fsync) the migration-completed marker. The content is
/// irrelevant — only the file's durable existence is the signal — so a single
/// byte keeps it minimal. Atomicity + fsync guarantee that once this returns
/// `Ok`, the marker survives a crash, which is what makes deleting the legacy
/// file afterward safe.
fn write_migration_marker(marker_path: &std::path::Path) -> Result<(), String> {
    use atomic_write_file::AtomicWriteFile;

    let mut file = AtomicWriteFile::open(marker_path)
        .map_err(|e| format!("open migration marker for atomic write: {e}"))?;
    file.write_all(b"1")
        .map_err(|e| format!("write migration marker: {e}"))?;
    file.commit()
        .map_err(|e| format!("commit migration marker: {e}"))
}

/// Which backend `persist_identity` wrote to. The caller writes the migration
/// marker only after a keyring success — on the file-fallback arm the key is on
/// disk and a marker would wrongly trip the next Unreachable boot into failing
/// closed.
enum PersistBackend {
    Keyring,
    File,
}

/// Generate a fresh identity, persist it through the store, return it.
///
/// On a keyring-backed persist no file is written, so a later
/// keyring-Unreachable boot would see "no file, no marker" (identical to a
/// fresh install) and silently rotate the identity. Writing the marker here
/// makes that boot fail closed. If the marker write fails, fall back to the
/// `0o600` file so the key is never keyring-only-without-marker.
fn generate_and_persist(
    store: &impl IdentityKeyStore,
    legacy_path: &std::path::Path,
    data_dir: &std::path::Path,
) -> Result<Keys, String> {
    let keys = Keys::generate();
    if let PersistBackend::Keyring = persist_identity(store, &keys, legacy_path)? {
        let marker_path = migration_marker_path(data_dir);
        if let Err(e) = write_migration_marker(&marker_path) {
            eprintln!(
                "buzz-desktop: stored identity in keyring but failed to write migration marker \
                 ({e}); saving identity.key fallback so the key is not stranded"
            );
            save_key_file(legacy_path, &keys)?;
        }
    }
    eprintln!(
        "buzz-desktop: generated and saved identity pubkey {}",
        keys.public_key().to_hex()
    );
    Ok(keys)
}

/// Persist `keys` through the store, falling back to the `0o600` file when the
/// keyring write fails on an availability error. Reports which backend held the
/// key so the caller can write the migration marker only on keyring success.
fn persist_identity(
    store: &impl IdentityKeyStore,
    keys: &Keys,
    legacy_path: &std::path::Path,
) -> Result<PersistBackend, String> {
    let nsec = keys
        .secret_key()
        .to_bech32()
        .map_err(|e| format!("encode nsec: {e}"))?;
    match store.store(IDENTITY_KEY_NAME, &nsec) {
        Ok(()) => Ok(PersistBackend::Keyring),
        Err(keyring_err) => {
            eprintln!("buzz-desktop: keyring write failed ({keyring_err}), using file fallback");
            save_key_file(legacy_path, keys)?;
            Ok(PersistBackend::File)
        }
    }
}

/// Best-effort removal of a leftover `identity.key` once the keyring is the
/// authoritative store. Idempotent: a missing file is success. Logs but does
/// not error on failure — a delete failure must never block startup.
fn cleanup_leftover_identity_file(legacy_path: &std::path::Path) {
    if !legacy_path.exists() {
        return;
    }
    match std::fs::remove_file(legacy_path) {
        Ok(()) => eprintln!("buzz-desktop: removed leftover identity.key (key is in keyring)"),
        Err(e) => eprintln!("buzz-desktop: failed to remove leftover identity.key: {e}"),
    }
}

/// Quarantine a corrupt `identity.key` with a timestamp so prior backups are
/// never overwritten.
fn quarantine_corrupt_key(key_path: &std::path::Path, data_dir: &std::path::Path, error: &str) {
    if !key_path.exists() {
        return;
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let bad_name = format!("identity.key.bad.{ts}");
    eprintln!("buzz-desktop: corrupt identity.key ({error}), quarantining to {bad_name}");
    let bad_path = data_dir.join(bad_name);
    if std::fs::rename(key_path, &bad_path).is_err() {
        let _ = std::fs::remove_file(key_path);
    }
}

fn load_key_file(path: &std::path::Path) -> Result<Keys, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read identity.key: {e}"))?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err("empty identity.key".to_string());
    }
    Keys::parse(trimmed).map_err(|e| format!("parse identity.key: {e}"))
}

/// Atomically write the key to disk. Uses `atomic-write-file` which:
/// 1. Writes to a temp file in the same directory
/// 2. Calls fsync on the file
/// 3. Renames temp → target (atomic on POSIX, best-effort on Windows)
/// 4. Calls fsync on the parent directory
///
/// On Unix, the file is created with mode 0600 (owner read/write only).
/// On Windows, default ACLs apply — the app data directory is already
/// per-user, so the key is not world-readable in practice.
pub(crate) fn save_key_file(path: &std::path::Path, keys: &Keys) -> Result<(), String> {
    use atomic_write_file::AtomicWriteFile;

    let nsec = keys
        .secret_key()
        .to_bech32()
        .map_err(|e| format!("encode nsec: {e}"))?;

    let mut file = AtomicWriteFile::open(path)
        .map_err(|e| format!("open identity.key for atomic write: {e}"))?;

    // Set owner-only permissions before writing the secret.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("set identity.key permissions: {e}"))?;
    }

    file.write_all(nsec.as_bytes())
        .map_err(|e| format!("write identity.key: {e}"))?;
    file.commit()
        .map_err(|e| format!("commit identity.key: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_key_eq(a: &Keys, b: &Keys) {
        assert_eq!(a.public_key().to_hex(), b.public_key().to_hex());
    }

    /// `BUZZ_PRIVATE_KEY` is process-global; serialize the env-mutating tests
    /// so they don't race each other under the parallel test runner.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `body` with `BUZZ_PRIVATE_KEY` set to `value` (or unset when `None`),
    /// restoring the prior value afterward.
    fn with_env_key<T>(value: Option<&str>, body: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var("BUZZ_PRIVATE_KEY").ok();
        match value {
            Some(v) => std::env::set_var("BUZZ_PRIVATE_KEY", v),
            None => std::env::remove_var("BUZZ_PRIVATE_KEY"),
        }
        let out = body();
        match prior {
            Some(v) => std::env::set_var("BUZZ_PRIVATE_KEY", v),
            None => std::env::remove_var("BUZZ_PRIVATE_KEY"),
        }
        out
    }

    #[test]
    fn identity_from_env_wins_when_valid() {
        let configured = Keys::generate();
        let nsec = configured.secret_key().to_bech32().unwrap();

        let resolved =
            with_env_key(Some(&nsec), identity_from_env).expect("valid env key must resolve");

        assert_key_eq(&configured, &resolved);
    }

    #[test]
    fn identity_from_env_none_when_absent() {
        assert!(with_env_key(None, identity_from_env).is_none());
    }

    #[test]
    fn identity_from_env_none_when_malformed() {
        // A malformed env var falls through to persisted resolution rather than
        // winning — otherwise a typo'd key would silently shadow the real one.
        assert!(with_env_key(Some("not-a-valid-nsec"), identity_from_env).is_none());
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        let keys = Keys::generate();

        save_key_file(&path, &keys).unwrap();
        let loaded = load_key_file(&path).unwrap();
        assert_key_eq(&keys, &loaded);
    }

    #[test]
    fn load_rejects_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        std::fs::write(&path, "").unwrap();

        assert!(load_key_file(&path).is_err());
    }

    #[test]
    fn load_rejects_corrupt_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        std::fs::write(&path, "not-a-valid-nsec").unwrap();

        assert!(load_key_file(&path).is_err());
    }

    #[test]
    fn load_missing_file_is_err() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.key");

        assert!(load_key_file(&path).is_err());
    }

    #[test]
    fn cleanup_removes_leftover_identity_file() {
        // Item 1: a leftover identity.key (from a migration whose remove_file
        // failed) is deleted once the keyring is authoritative, so plaintext
        // does not linger on disk.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        save_key_file(&path, &Keys::generate()).unwrap();
        assert!(path.exists());

        cleanup_leftover_identity_file(&path);

        assert!(!path.exists());
    }

    #[test]
    fn cleanup_is_noop_when_no_leftover_file() {
        // Idempotent: the cleanup runs on every keyring-Present boot, so a
        // missing file must be a silent success, not an error or panic.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        assert!(!path.exists());

        cleanup_leftover_identity_file(&path);

        assert!(!path.exists());
    }

    #[test]
    fn save_creates_file_with_valid_nsec() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        let keys = Keys::generate();

        save_key_file(&path, &keys).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("nsec1"));
    }

    #[cfg(unix)]
    #[test]
    fn save_creates_file_with_restricted_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        let keys = Keys::generate();

        save_key_file(&path, &keys).unwrap();

        let perms = std::fs::metadata(&path).unwrap().permissions();
        assert_eq!(perms.mode() & 0o777, 0o600);
    }

    #[test]
    fn save_overwrites_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");

        let keys1 = Keys::generate();
        save_key_file(&path, &keys1).unwrap();

        let keys2 = Keys::generate();
        save_key_file(&path, &keys2).unwrap();

        let loaded = load_key_file(&path).unwrap();
        assert_key_eq(&keys2, &loaded);
    }

    use std::cell::RefCell;
    use std::collections::HashMap;

    use crate::secret_store::KeyringProbe;

    /// In-memory [`IdentityKeyStore`] for testing identity recovery without the
    /// OS keyring. Seeded with an initial value and a probe outcome; records
    /// every `delete`/`store` so tests can assert the keyring was cleared and
    /// rewritten. `write_and_verify` succeeds (store then load reflects it).
    struct FakeIdentityStore {
        probe: KeyringProbe,
        slot: RefCell<HashMap<String, String>>,
        deleted: RefCell<Vec<String>>,
        /// When true, `store` returns an availability error, driving the
        /// keyring-write-failure → file-fallback arm of `persist_identity`.
        store_fails: bool,
    }

    impl FakeIdentityStore {
        fn present_with(value: &str) -> Self {
            let mut slot = HashMap::new();
            slot.insert(IDENTITY_KEY_NAME.to_string(), value.to_string());
            Self {
                probe: KeyringProbe::Present,
                slot: RefCell::new(slot),
                deleted: RefCell::new(Vec::new()),
                store_fails: false,
            }
        }

        /// Backend down this boot: probe is `Unreachable` and the slot is empty
        /// (the real key, if any, is in the keyring we cannot reach).
        fn unreachable() -> Self {
            Self {
                probe: KeyringProbe::Unreachable,
                slot: RefCell::new(HashMap::new()),
                deleted: RefCell::new(Vec::new()),
                store_fails: false,
            }
        }

        /// Backend reachable with no entry — drives the one-time migration path.
        /// `store`/`load` go through the slot, so a read-back verify succeeds.
        fn reachable_but_empty() -> Self {
            Self {
                probe: KeyringProbe::ReachableButEmpty,
                slot: RefCell::new(HashMap::new()),
                deleted: RefCell::new(Vec::new()),
                store_fails: false,
            }
        }

        /// Reachable-but-empty probe whose `store` always fails — exercises the
        /// keyring-write-failure → `0o600` file-fallback arm.
        fn store_failing() -> Self {
            Self {
                probe: KeyringProbe::ReachableButEmpty,
                slot: RefCell::new(HashMap::new()),
                deleted: RefCell::new(Vec::new()),
                store_fails: true,
            }
        }
    }

    impl IdentityKeyStore for FakeIdentityStore {
        fn probe(&self, _name: &str) -> KeyringProbe {
            self.probe
        }
        fn load(&self, name: &str) -> Result<Option<String>, String> {
            Ok(self.slot.borrow().get(name).cloned())
        }
        fn store(&self, name: &str, value: &str) -> Result<(), String> {
            if self.store_fails {
                return Err("simulated keyring write failure".to_string());
            }
            self.slot
                .borrow_mut()
                .insert(name.to_string(), value.to_string());
            Ok(())
        }
        fn delete(&self, name: &str) -> Result<(), String> {
            self.deleted.borrow_mut().push(name.to_string());
            self.slot.borrow_mut().remove(name);
            Ok(())
        }
    }

    #[test]
    fn corrupt_keyring_recovers_valid_file_without_rotating() {
        // The load-bearing regression guard. When the keyring holds a corrupt
        // nsec (Present) AND a valid `identity.key` is on disk (leftover from a
        // failed prior migration), recovery must RECOVER THE FILE'S identity —
        // not quarantine the file and rotate to a fresh key (the original
        // hazard). The corrupt keyring value must be cleared and replaced by the
        // file's key (migrated in).
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");
        let file_keys = Keys::generate();
        save_key_file(&legacy_path, &file_keys).unwrap();

        let store = FakeIdentityStore::present_with("not-a-valid-nsec");
        let resolved = resolve_identity_with_store(&store, &legacy_path, dir.path()).unwrap();

        // The FILE's identity is recovered — NOT a freshly generated one.
        assert_key_eq(&file_keys, &resolved.keys);
        // The corrupt keyring value was cleared.
        assert_eq!(store.deleted.borrow().as_slice(), [IDENTITY_KEY_NAME]);
        // The keyring now holds the file's key (migrated in, read-back verified).
        let file_nsec = file_keys.secret_key().to_bech32().unwrap();
        assert_eq!(
            store
                .slot
                .borrow()
                .get(IDENTITY_KEY_NAME)
                .map(String::as_str),
            Some(file_nsec.as_str())
        );
        // The valid file was migrated (deleted), not quarantined to .bad.*.
        assert!(!legacy_path.exists());
        assert!(std::fs::read_dir(dir.path()).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".bad.")));
    }

    #[test]
    fn corrupt_keyring_generates_fresh_only_when_no_file() {
        // With a corrupt keyring value and NO file on disk, generate-fresh is
        // the correct last resort — and the corrupt keyring value is cleared
        // first.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");
        assert!(!legacy_path.exists());

        let store = FakeIdentityStore::present_with("not-a-valid-nsec");
        let resolved = resolve_identity_with_store(&store, &legacy_path, dir.path()).unwrap();

        assert_eq!(store.deleted.borrow().as_slice(), [IDENTITY_KEY_NAME]);
        // A fresh, valid key was persisted to the keyring (replacing the cleared
        // corrupt value).
        let stored = store.slot.borrow().get(IDENTITY_KEY_NAME).cloned();
        assert_eq!(
            stored.as_deref(),
            Some(resolved.keys.secret_key().to_bech32().unwrap().as_str())
        );
    }

    #[test]
    fn valid_keyring_is_used_and_matching_leftover_file_cleaned_up() {
        // A valid keyring entry and a leftover identity.key with the SAME pubkey
        // (stale leftover from a migration whose remove_file previously failed):
        // keyring wins, plaintext is removed without adoption.
        let keyring_keys = Keys::generate();
        let nsec = keyring_keys.secret_key().to_bech32().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");
        // Same key in file as keyring → stale leftover, not an import.
        save_key_file(&legacy_path, &keyring_keys).unwrap();

        let store = FakeIdentityStore::present_with(&nsec);
        let resolved = resolve_identity_with_store(&store, &legacy_path, dir.path()).unwrap();

        assert_key_eq(&keyring_keys, &resolved.keys);
        assert!(!resolved.lost);
        assert!(store.deleted.borrow().is_empty());
        assert!(!legacy_path.exists());
    }

    #[test]
    fn unreachable_post_migration_fails_closed_when_marker_present() {
        // The silent-rotation hazard (Wes Comment 1). After a migration the
        // file is gone and the marker exists; a later boot with the keyring
        // unreachable must FAIL CLOSED — the real key is in the keyring, not
        // gone, so generating a fresh one would silently rotate the identity.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");
        write_migration_marker(&migration_marker_path(dir.path())).unwrap();
        assert!(!legacy_path.exists());

        let store = FakeIdentityStore::unreachable();
        let result = resolve_identity_with_store(&store, &legacy_path, dir.path());

        assert!(
            result.is_err(),
            "must fail closed, not generate a fresh key"
        );
        // No identity file was written — nothing was generated or persisted.
        assert!(!legacy_path.exists());
    }

    #[test]
    fn unreachable_first_run_generates_to_file_when_no_marker() {
        // Genuine first-EVER launch on a machine whose keyring is down: no file,
        // no marker. There is no prior identity to protect, so generating to the
        // `0o600` file is correct — fail-closed here would block a legitimate
        // first launch.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");
        assert!(!legacy_path.exists());
        assert!(!migration_marker_path(dir.path()).exists());

        let store = FakeIdentityStore::unreachable();
        let resolved = resolve_identity_with_store(&store, &legacy_path, dir.path()).unwrap();

        // A fresh key was generated and persisted to the file (keyring is down).
        let from_file = load_key_file(&legacy_path).unwrap();
        assert_key_eq(&resolved.keys, &from_file);
    }

    #[test]
    fn migration_writes_marker_before_deleting_file() {
        // Crash-safe ordering: a successful migration must leave the marker on
        // disk AND remove the file. The marker existing while the file is gone
        // is the durable post-migration signal the Unreachable arm relies on;
        // "file gone, no marker" must never be the resting state.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");
        let file_keys = Keys::generate();
        save_key_file(&legacy_path, &file_keys).unwrap();

        // ReachableButEmpty drives the one-time migration path.
        let store = FakeIdentityStore::reachable_but_empty();
        let resolved = resolve_identity_with_store(&store, &legacy_path, dir.path()).unwrap();

        assert_key_eq(&file_keys, &resolved.keys);
        // Marker written, file deleted — the safe resting state.
        assert!(migration_marker_path(dir.path()).exists());
        assert!(!legacy_path.exists());
    }

    #[test]
    fn fresh_keyring_generate_writes_marker() {
        // Fix 1 (Pinky comment 1): a fresh install generating straight into a
        // reachable-but-empty keyring must write the marker. Without it, "no
        // file, no marker" matches a never-launched machine, so a later
        // Unreachable boot would silently rotate the key.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");
        assert!(!legacy_path.exists());

        let store = FakeIdentityStore::reachable_but_empty();
        let resolved = resolve_identity_with_store(&store, &legacy_path, dir.path()).unwrap();

        // The key was stored in the keyring (not the file), and the marker marks it.
        assert!(!legacy_path.exists());
        assert!(migration_marker_path(dir.path()).exists());
        assert_eq!(
            store
                .slot
                .borrow()
                .get(IDENTITY_KEY_NAME)
                .map(String::as_str),
            Some(resolved.keys.secret_key().to_bech32().unwrap().as_str())
        );
    }

    #[test]
    fn fresh_keyring_generate_then_unreachable_fails_closed() {
        // The end-to-end guard for Fix 1: after a fresh keyring-created identity
        // (marker written, no file), a later boot with the keyring unreachable
        // must FAIL CLOSED rather than generate a new key and rotate identity.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");

        // First boot: fresh generate into a reachable keyring.
        let reachable = FakeIdentityStore::reachable_but_empty();
        resolve_identity_with_store(&reachable, &legacy_path, dir.path()).unwrap();
        assert!(!legacy_path.exists());
        assert!(migration_marker_path(dir.path()).exists());

        // Second boot: keyring is down. No file + marker present → fail closed.
        let unreachable = FakeIdentityStore::unreachable();
        let result = resolve_identity_with_store(&unreachable, &legacy_path, dir.path());

        assert!(
            result.is_err(),
            "must fail closed, not generate a fresh key"
        );
        assert!(!legacy_path.exists());
    }

    #[test]
    fn fresh_generate_keyring_failure_falls_back_to_file_without_marker() {
        // Fix 1 correctness on the file-fallback arm: when the keyring write
        // FAILS during a fresh generate, the key must land in the `0o600` file
        // and the marker must NOT be written — a marker here would wrongly trip
        // the next Unreachable boot into failing closed even though the key is
        // sitting in the file.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");

        let store = FakeIdentityStore::store_failing();
        let resolved = resolve_identity_with_store(&store, &legacy_path, dir.path()).unwrap();

        // Key persisted to the file (fallback), and recoverable from it.
        let from_file = load_key_file(&legacy_path).unwrap();
        assert_key_eq(&resolved.keys, &from_file);
        // No marker: the file is the authoritative store, not the keyring.
        assert!(!migration_marker_path(dir.path()).exists());
    }

    // ── New tests for the three defects fixed in this PR ─────────────────────

    #[test]
    fn import_persists_to_keyring_reboot_resolves_imported_pubkey() {
        // (a) import persists to keyring → simulated reboot resolves the
        // imported pubkey.
        //
        // `persist_identity_to_keyring` is the kernel called by
        // `import_identity`. After it succeeds the keyring slot holds the
        // imported nsec. A fresh store seeded with that nsec (simulating a
        // reboot where the keyring has the value) must resolve to the same key.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");
        let imported_keys = Keys::generate();

        // Simulate what import_identity does: persist to keyring.
        let store_import = FakeIdentityStore::reachable_but_empty();
        persist_identity_to_keyring(&store_import, &imported_keys, &legacy_path, dir.path())
            .expect("persist_identity_to_keyring must succeed with a reachable store");

        // Keyring slot now holds the imported nsec.
        let stored_nsec = store_import
            .slot
            .borrow()
            .get(IDENTITY_KEY_NAME)
            .cloned()
            .expect("keyring must hold the imported nsec after persist");
        assert_eq!(stored_nsec, imported_keys.secret_key().to_bech32().unwrap());

        // Simulated reboot: new store with Present probe, seeded with the stored nsec.
        let store_reboot = FakeIdentityStore::present_with(&stored_nsec);
        let resolved =
            resolve_identity_with_store(&store_reboot, &legacy_path, dir.path()).unwrap();

        // The resolved key is the imported one — identity survives the reboot.
        assert_key_eq(&imported_keys, &resolved.keys);
        assert!(!resolved.lost);
        // No identity.key left on disk (was deleted by persist_identity_to_keyring).
        assert!(!legacy_path.exists());
    }

    #[test]
    fn present_keyring_with_mismatched_file_adopts_file_key() {
        // (b) Present + mismatched identity.key → file's key adopted into
        // keyring, no data loss, file removed.
        //
        // This auto-heals installs already stuck in the re-onboarding loop:
        // the keyring holds the shadow key generated at first launch, while
        // identity.key holds the user's imported key from a subsequent import
        // that only reached the file (pre-fix bug). Resolution must adopt the
        // file's key as the user's explicit intent.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");

        let keyring_keys = Keys::generate();
        let keyring_nsec = keyring_keys.secret_key().to_bech32().unwrap();

        // identity.key has a DIFFERENT key — the user's import.
        let file_keys = Keys::generate();
        save_key_file(&legacy_path, &file_keys).unwrap();

        let store = FakeIdentityStore::present_with(&keyring_nsec);
        let resolved = resolve_identity_with_store(&store, &legacy_path, dir.path()).unwrap();

        // The file's key (user's explicit import) wins.
        assert_key_eq(&file_keys, &resolved.keys);
        assert!(!resolved.lost);

        // The keyring now holds the file's key (overwritten with read-back verify).
        let file_nsec = file_keys.secret_key().to_bech32().unwrap();
        assert_eq!(
            store
                .slot
                .borrow()
                .get(IDENTITY_KEY_NAME)
                .map(String::as_str),
            Some(file_nsec.as_str())
        );

        // identity.key was removed after adoption.
        assert!(!legacy_path.exists());
    }

    #[test]
    fn reachable_but_empty_with_marker_and_no_file_returns_lost() {
        // (d) ReachableButEmpty + marker + no file → "lost" state, NO new key
        // generated into the keyring.
        //
        // The marker says a key was once stored in the keyring. If the keyring
        // is now empty (entry deleted externally, new OS login session cleared
        // it, etc.) and there is no file fallback, the user's key is truly
        // gone. Resolution must NOT silently generate a new identity; it must
        // surface a "lost" state so the frontend can prompt re-import.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");
        // Write the migration marker — a key was once in the keyring.
        write_migration_marker(&migration_marker_path(dir.path())).unwrap();
        assert!(!legacy_path.exists()); // no file fallback

        let store = FakeIdentityStore::reachable_but_empty();
        let resolved = resolve_identity_with_store(&store, &legacy_path, dir.path()).unwrap();

        // The "lost" flag is set — the frontend must prompt re-import.
        assert!(resolved.lost, "identity lost state must be surfaced");

        // No key was persisted to the keyring — the ephemeral key is in-memory
        // only and must not overwrite the user's actual (externally lost) key.
        assert!(
            store.slot.borrow().is_empty(),
            "no key must be written to keyring when identity is lost"
        );

        // No identity.key written either — the ephemeral key is transient.
        assert!(!legacy_path.exists());
    }

    #[test]
    fn import_keyring_failure_falls_back_to_file_so_key_is_recoverable() {
        // (e) keyring-write failure during import → file fallback survives so
        // the key is recoverable on next boot.
        //
        // When the keyring is unavailable, `persist_identity_to_keyring`
        // returns Err and the caller writes identity.key as a fallback. No
        // migration marker must be written — a marker here would wrongly trip
        // the Unreachable-boot fail-closed check even though the key is in the
        // file.
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = dir.path().join("identity.key");
        let imported_keys = Keys::generate();

        // Keyring write always fails.
        let store = FakeIdentityStore::store_failing();
        let result = persist_identity_to_keyring(&store, &imported_keys, &legacy_path, dir.path());

        assert!(
            result.is_err(),
            "persist_identity_to_keyring must propagate keyring failure"
        );

        // Caller falls back: write the file so the key survives.
        save_key_file(&legacy_path, &imported_keys).unwrap();

        // Key is recoverable from the file on next boot.
        let from_file = load_key_file(&legacy_path).unwrap();
        assert_key_eq(&imported_keys, &from_file);

        // No marker written — the file is the authoritative store, not the
        // keyring, and a marker would cause fail-closed on a later Unreachable
        // boot.
        assert!(!migration_marker_path(dir.path()).exists());
    }
}
