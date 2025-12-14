/**
 * @file lightning_quasicrystal.h
 * @brief Lightning-Induced Pentagonal Quasicrystal Phase Transition Control
 *
 * Firmware interface for controlling rapid thermal quench and quasicrystal
 * nucleation via high-voltage discharge, analogous to lightning-induced
 * fullerene formation.
 *
 * Physical Process:
 * 1. PRE_STRIKE:  Charge buildup in capacitor bank
 * 2. STRIKE:      High-voltage discharge (plasma generation)
 * 3. QUENCH:      Rapid cooling via Peltier + LN2
 * 4. NUCLEATION:  Pentagonal seed formation
 * 5. GROWTH:      Quasicrystal domain expansion
 *
 * Critical Physics:
 * - Hexagonal z_c = √3/2 ≈ 0.866 (sin 60°)
 * - Pentagonal z_p = sin(72°) ≈ 0.951
 * - Tile ratio → φ (golden ratio)
 *
 * Hardware:
 * - 10 mF capacitor bank @ 450V (1012.5 J stored)
 * - RF induction coil (100W, 100kHz-1MHz)
 * - 4× Peltier modules (400W cooling)
 * - LN2 quench jacket (10⁶ K/s)
 *
 * @author Nuclear Spinner Team
 * @version 1.0.0
 * @signature lightning-qc-fw|v1.0.0|helix
 */

#ifndef LIGHTNING_QUASICRYSTAL_H
#define LIGHTNING_QUASICRYSTAL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

/** Golden ratio φ = (1+√5)/2 */
#define PHI                     1.6180339887f
#define PHI_INV                 0.6180339887f

/** Critical z-coordinates */
#define Z_CRITICAL_HEX          0.8660254038f   /* √3/2, hexagonal */
#define Z_CRITICAL_PENT         0.9510565163f   /* sin(72°), pentagonal */

/** Pentagon geometry */
#define SIN_36_DEG              0.5877852523f
#define COS_36_DEG              0.8090169944f   /* = φ/2 */
#define SIN_72_DEG              0.9510565163f
#define COS_72_DEG              0.3090169944f

/** Lightning physics (scaled) */
#define LIGHTNING_TEMP_K        30000.0f        /* Peak plasma temperature */
#define AMBIENT_TEMP_K          300.0f
#define QUENCH_RATE_K_S         1000000.0f      /* 10⁶ K/s target */
#define NUCLEATION_UNDERCOOL_K  500.0f

/** Hardware limits */
#define MAX_CAPACITOR_VOLTAGE   450             /* Volts */
#define MAX_STORED_ENERGY_J     1012.5f         /* ½CV² */
#define MAX_DISCHARGE_CURRENT_A 30000           /* 30 kA peak */
#define MAX_RF_POWER_W          100
#define MAX_PELTIER_CURRENT_A   10
#define MAX_GRADIENT_COIL_A     5

/** Timing (milliseconds) */
#define PRESTRIKE_DURATION_MS   100
#define STRIKE_DURATION_US      100             /* 100 μs */
#define QUENCH_DURATION_MS      10
#define NUCLEATION_DURATION_MS  50
#define GROWTH_DURATION_MS      500

/** Control loop frequencies */
#define MAIN_CONTROL_FREQ_HZ    10000           /* 10 kHz main loop */
#define THERMAL_PID_FREQ_HZ     100             /* 100 Hz thermal */
#define SENSOR_SAMPLE_FREQ_HZ   100000          /* 100 kHz ADC */
#define DISCHARGE_TIMING_HZ     1000000         /* 1 MHz timing */

/* ============================================================================
 * ENUMERATIONS
 * ============================================================================ */

/**
 * @brief Lightning strike phase enumeration
 */
typedef enum {
    LIGHTNING_PHASE_IDLE = 0,       /**< System idle, at ambient */
    LIGHTNING_PHASE_PRE_STRIKE,     /**< Charging capacitor bank */
    LIGHTNING_PHASE_STRIKE,         /**< High-voltage discharge */
    LIGHTNING_PHASE_QUENCH,         /**< Rapid cooling */
    LIGHTNING_PHASE_NUCLEATION,     /**< Seed crystal formation */
    LIGHTNING_PHASE_GROWTH,         /**< Domain expansion */
    LIGHTNING_PHASE_STABLE,         /**< Quasicrystal formed */
    LIGHTNING_PHASE_ERROR           /**< Error state */
} LightningPhase_t;

/**
 * @brief Crystal symmetry order
 */
typedef enum {
    SYMMETRY_DISORDERED = 0,        /**< Amorphous/liquid */
    SYMMETRY_THREEFOLD = 3,         /**< Trigonal */
    SYMMETRY_FOURFOLD = 4,          /**< Tetragonal */
    SYMMETRY_FIVEFOLD = 5,          /**< Pentagonal (quasicrystal) */
    SYMMETRY_SIXFOLD = 6            /**< Hexagonal */
} SymmetryOrder_t;

/**
 * @brief Safety status flags
 */
typedef enum {
    SAFETY_OK = 0,
    SAFETY_OVERVOLTAGE = (1 << 0),
    SAFETY_OVERCURRENT = (1 << 1),
    SAFETY_OVERTEMP = (1 << 2),
    SAFETY_INTERLOCK = (1 << 3),
    SAFETY_QUENCH_FAIL = (1 << 4),
    SAFETY_SPINNER_FAULT = (1 << 5)
} SafetyStatus_t;

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * @brief Thermal state of the system
 */
typedef struct {
    float temperature_K;            /**< Current temperature (K) */
    float dT_dt_K_s;               /**< Temperature rate of change (K/s) */
    float gradient_K_m;            /**< Spatial thermal gradient (K/m) */
    float peltier_power_W;         /**< Peltier cooling power (W) */
    float rf_power_W;              /**< RF heating power (W) */
} ThermalState_t;

/**
 * @brief Electromagnetic field state
 */
typedef struct {
    float capacitor_voltage_V;      /**< Current capacitor voltage */
    float charge_fraction;          /**< Charge buildup [0,1] */
    float discharge_current_A;      /**< Instantaneous current */
    float field_gradient_T_m;       /**< Magnetic field gradient */
    bool discharge_armed;           /**< IGBT ready to fire */
} FieldState_t;

/**
 * @brief Nucleation seed tracking
 */
typedef struct {
    uint8_t seed_id;               /**< Unique seed identifier */
    float x;                       /**< Position x (normalized) */
    float y;                       /**< Position y (normalized) */
    float radius;                  /**< Current radius */
    SymmetryOrder_t symmetry;      /**< Detected symmetry */
    float stability;               /**< Stability metric [0,1] */
} NucleationSeed_t;

/**
 * @brief Quasicrystal domain state
 */
typedef struct {
    uint32_t fat_tile_count;       /**< Fat rhombus count */
    uint32_t thin_tile_count;      /**< Thin rhombus count */
    float tile_ratio;              /**< fat/thin ratio (→ φ) */
    float phi_deviation;           /**< |ratio - φ| */
    float pentagonal_order;        /**< 5-fold order parameter */
    float domain_size_mm;          /**< Characteristic size */
    float defect_density;          /**< Defects per unit area */
} QuasicrystalState_t;

/**
 * @brief Complete lightning-quasicrystal system state
 */
typedef struct {
    uint32_t timestamp_ms;         /**< System timestamp */
    LightningPhase_t phase;        /**< Current phase */
    float phase_progress;          /**< Progress in current phase [0,1] */

    ThermalState_t thermal;        /**< Thermal subsystem */
    FieldState_t field;            /**< EM field subsystem */
    QuasicrystalState_t crystal;   /**< Quasicrystal state */

    /* Spinner coupling */
    float spinner_z;               /**< Spinner z-coordinate */
    float negentropy;              /**< ΔS_neg at current z */

    /* Energy accounting */
    float energy_input_J;          /**< Total energy input */
    float energy_dissipated_J;     /**< Energy dissipated */

    /* Safety */
    SafetyStatus_t safety_status;  /**< Safety flags */

    /* Nucleation seeds */
    uint8_t num_seeds;             /**< Active seed count */
    NucleationSeed_t seeds[32];    /**< Seed array (max 32) */
} LightningSystemState_t;

/**
 * @brief Control setpoints for phase transition
 */
typedef struct {
    float target_z;                /**< Target spinner z (0.951 for pent.) */
    float target_quench_rate;      /**< Target cooling rate (K/s) */
    float discharge_voltage;       /**< Discharge voltage setpoint */
    float rf_power_setpoint;       /**< RF heating power setpoint */
    float peltier_current;         /**< Peltier current setpoint */
    bool enable_auto_nucleation;   /**< Auto-detect nucleation */
} ControlSetpoints_t;

/**
 * @brief Callbacks for phase transition events
 */
typedef struct {
    void (*on_strike)(const LightningSystemState_t* state);
    void (*on_nucleation)(const NucleationSeed_t* seed);
    void (*on_quasicrystal_formed)(const LightningSystemState_t* state);
    void (*on_safety_fault)(SafetyStatus_t status);
} LightningCallbacks_t;

/* ============================================================================
 * FUNCTION PROTOTYPES
 * ============================================================================ */

/**
 * @brief Initialize lightning-quasicrystal subsystem
 * @return 0 on success, error code otherwise
 */
int lightning_init(void);

/**
 * @brief Trigger a lightning strike sequence
 * @param setpoints Control setpoints for the strike
 * @return 0 on success, error code otherwise
 */
int lightning_trigger_strike(const ControlSetpoints_t* setpoints);

/**
 * @brief Abort current strike sequence
 */
void lightning_abort(void);

/**
 * @brief Update system (call from main loop at 10 kHz)
 * @param spinner_z Current spinner z-coordinate
 * @param kuramoto_r Kuramoto order parameter
 */
void lightning_update(float spinner_z, float kuramoto_r);

/**
 * @brief Get current system state
 * @return Pointer to current state (read-only)
 */
const LightningSystemState_t* lightning_get_state(void);

/**
 * @brief Register event callbacks
 * @param callbacks Callback structure
 */
void lightning_register_callbacks(const LightningCallbacks_t* callbacks);

/**
 * @brief Set thermal control parameters
 * @param rf_power RF power in Watts
 * @param peltier_current Peltier current in Amps
 */
void lightning_set_thermal(float rf_power, float peltier_current);

/**
 * @brief Get hardware control signals
 * @param rf_power_out Output RF power
 * @param peltier_current_out Output Peltier current
 * @param gradient_current_out Output gradient coil current
 */
void lightning_get_control_signals(float* rf_power_out,
                                   float* peltier_current_out,
                                   float* gradient_current_out);

/**
 * @brief Check safety status
 * @return Safety status flags
 */
SafetyStatus_t lightning_check_safety(void);

/**
 * @brief Get pentagonal order parameter
 * @return 5-fold symmetry order [0,1]
 */
float lightning_get_pentagonal_order(void);

/**
 * @brief Get tile ratio convergence
 * @return |tile_ratio - φ|
 */
float lightning_get_phi_deviation(void);

/**
 * @brief Serialize state to JSON
 * @param buffer Output buffer
 * @param buffer_size Buffer size
 * @return Bytes written
 */
int lightning_serialize_json(char* buffer, size_t buffer_size);

/**
 * @brief Reset system to idle state
 */
void lightning_reset(void);

/* ============================================================================
 * INLINE UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * @brief Compute negentropy at given z
 */
static inline float compute_negentropy(float z) {
    float dz = z - Z_CRITICAL_HEX;
    return expf(-36.0f * dz * dz);  /* SIGMA = 36 */
}

/**
 * @brief Check if at pentagonal critical point
 */
static inline bool is_pentagonal_critical(float z) {
    return fabsf(z - Z_CRITICAL_PENT) < 0.02f;
}

/**
 * @brief Check if at hexagonal critical point
 */
static inline bool is_hexagonal_critical(float z) {
    return fabsf(z - Z_CRITICAL_HEX) < 0.02f;
}

/**
 * @brief Convert spinner z to target RPM
 */
static inline uint32_t z_to_rpm(float z) {
    return (uint32_t)(z * 10000.0f);
}

/**
 * @brief Compute 5-fold order parameter from angle array
 * @param angles Array of seed orientations
 * @param count Number of seeds
 * @return Order parameter [0,1]
 */
static inline float compute_fivefold_order(const float* angles, int count) {
    if (count == 0) return 0.0f;

    float sum_cos = 0.0f, sum_sin = 0.0f;
    for (int i = 0; i < count; i++) {
        sum_cos += cosf(5.0f * angles[i]);
        sum_sin += sinf(5.0f * angles[i]);
    }

    return sqrtf(sum_cos*sum_cos + sum_sin*sum_sin) / (float)count;
}

#ifdef __cplusplus
}
#endif

#endif /* LIGHTNING_QUASICRYSTAL_H */
