       IDENTIFICATION DIVISION.
       PROGRAM-ID. ANISODIFF-C.
      *> COBOL port of the pygeoflood Perona-Malik anisotropic diffusion
      *> kernel (build spec D1, art_QEojt4gd).
      *>
      *> Operation order matches the frozen NumPy reference
      *> (tests/reference_anisodiff.py, frozen at d7bbd46) element for
      *> element, with the same boundary sentinels and directional NaN
      *> masking as the merged Rust port (rust/src/diffusion.rs).
      *>
      *> ARITHMETIC PRECISION - GnuCOBOL 3.2 note. COMPUTE lowers to the
      *> cob_decimal runtime, which bit-exactly rounds FINITE float
      *> results (verified element-wise against the reference across the
      *> full parity matrix) but decimalizes non-finite operands to zero,
      *> to zero, decimalizes non-finite results of numeric MOVEs, and
      *> produces undefined values for zero divisors. Even mixed-type
      *> comparisons (`x < 0 - MAX`) drop into decimal and corrupt
      *> non-finite operands. The kernel therefore:
      *>   - runs every arithmetic op that can see a non-finite value
      *>     through the F32-*/F64-* wrappers, which implement IEEE 754
      *>     special-case algebra explicitly (NaN by self-inequality,
      *>     infinity by range comparison against native COMP constants,
      *>     hex-constructed NaN/inf constants) and use COMPUTE only
      *>     where both operands are finite;
      *>   - widens f32->f64 and narrows f64->f32 through F64-OF-F32 /
      *>     F32-OF-F64 instead of numeric MOVEs;
      *>   - keeps all condition comparisons COMP-to-COMP (no embedded
      *>     arithmetic expressions).
      *> Finite overflow (squaring 2e30, say) rounds to +inf correctly
      *> the decimal path and needs no wrapper.
      *>
      *> Stencil state is COMP-1 (float32), the pipeline dtype. COMP-2
      *> paths exist only where NumPy promotes an explicit f64 scalar,
      *> gated by the PREC bit flags (mirror of the Rust `precision`
      *> tuple): bit 1 kappa, bit 2 south step, bit 4 east step, bit 8
      *> gamma. Flux, divergence, and the update run in the precision
      *> NumPy used at each step, with f32 narrowings exactly where the
      *> reference stores into float32 arrays.
      *>
      *> Called through shim.c, which owns the one-time GnuCOBOL runtime
      *> startup (cob_init). Pure function of its buffers: reads L-IN,
      *> writes L-SCRATCH and L-OUT, never L-IN. Single-threaded by
      *> construction; benchmark provenance records this so timings are
      *> read correctly.

       DATA DIVISION.
       WORKING-STORAGE SECTION.

      *> IEEE-754 bit constants. Little-endian byte patterns through
      *> alphanumeric hex literals, redefined as floating point. f32:
      *> NaN = 0x7FC00000 (quiet), +inf = 0x7F800000, -inf = 0xFF800000.
      *> f64: NaN = 0x7FF8000000000000, +inf = 0x7FF0000000000000,
      *> -inf = 0xFFF0000000000000.
       01 WS-NAN-X         PIC X(4) VALUE X"0000C07F".
       01 WS-NAN REDEFINES WS-NAN-X USAGE COMP-1.
       01 WS-PINF-X        PIC X(4) VALUE X"0000807F".
       01 WS-PINF REDEFINES WS-PINF-X USAGE COMP-1.
       01 WS-NINF-X        PIC X(4) VALUE X"000080FF".
       01 WS-NINF REDEFINES WS-NINF-X USAGE COMP-1.
       01 WS-NAN2-X        PIC X(8) VALUE X"000000000000F87F".
       01 WS-NAN2 REDEFINES WS-NAN2-X USAGE COMP-2.
       01 WS-PINF2-X       PIC X(8) VALUE X"000000000000F07F".
       01 WS-PINF2 REDEFINES WS-PINF2-X USAGE COMP-2.
       01 WS-NINF2-X       PIC X(8) VALUE X"000000000000F0FF".
       01 WS-NINF2 REDEFINES WS-NINF2-X USAGE COMP-2.
      *> Range bounds. Comparisons must be COMP-to-COMP: embedding
      *> arithmetic like `0 - MAX` in a condition drops the evaluation
      *> into the decimal runtime, which decimalizes non-finite operands
      *> before comparing. Only infinities compare outside the maxima.
       01 WS-FLT-MAX       USAGE COMP-1 VALUE 3.4028235E38.
       01 WS-NEG-MAX       USAGE COMP-1 VALUE -3.4028235E38.
       01 WS-DBL-MAX       USAGE COMP-2 VALUE 1.7976931348623157E308.
       01 WS-NEG-DBL-MAX   USAGE COMP-2 VALUE -1.7976931348623157E308.
      *> Smallest f64 that still narrows to finite f32 (values strictly
      *> above overflow to infinity, mirroring NumPy's f64->f32 cast).
       01 WS-F32-OVER      USAGE COMP-2 VALUE 3.4028236690202723E38.
       01 WS-NEG-F32-OVER  USAGE COMP-2 VALUE -3.4028236690202723E38.

       LOCAL-STORAGE SECTION.
      *> Loop counters and buffer indexes (1-based, row-major).
       01 LS-IT            USAGE BINARY-LONG.
       01 LS-I             USAGE BINARY-LONG.
       01 LS-J             USAGE BINARY-LONG.
       01 LS-IDX           USAGE BINARY-LONG.
       01 LS-PREC-REST     USAGE BINARY-LONG.

      *> Decoded PREC bitmask.
       01 LS-KAPPA-DOUBLE  USAGE BINARY-LONG.
       01 LS-SOUTH-DOUBLE  USAGE BINARY-LONG.
       01 LS-EAST-DOUBLE   USAGE BINARY-LONG.
       01 LS-GAMMA-DOUBLE  USAGE BINARY-LONG.

      *> Axis under evaluation, copied in before CALC-FLUX/DIVERGE.
       01 LS-A-STEP-D      USAGE COMP-2.
       01 LS-A-STEP-DOUBLE USAGE BINARY-LONG.
       01 LS-A-DOUBLE      USAGE BINARY-LONG.

      *> IEEE wrapper operand/result cells. f32 wrappers use WS-A/WS-B
      *> with the result in WS-R; f64 wrappers use WS-A2/WS-B2 with the
      *> result in WS-R2.
       01 WS-A             USAGE COMP-1.
       01 WS-B             USAGE COMP-1.
       01 WS-R             USAGE COMP-1.
       01 WS-A2            USAGE COMP-2.
       01 WS-B2            USAGE COMP-2.
       01 WS-R2            USAGE COMP-2.

      *> Flux chain: one rounding per temporary, matching NumPy's
      *> op-by-op semantics in the precision each step used.
       01 LS-FLUX-DELTA    USAGE COMP-1.
       01 LS-KAPPA-F       USAGE COMP-1.
       01 LS-STEP-F        USAGE COMP-1.
       01 LS-RATIO-F       USAGE COMP-1.
       01 LS-SQUARE-F      USAGE COMP-1.
       01 LS-TERM-F        USAGE COMP-1.
       01 LS-COEF-F        USAGE COMP-1.
       01 LS-FLUX-F        USAGE COMP-1.
      *> Double path (NumPy explicit-float64 scalar promotion).
       01 LS-RATIO-D       USAGE COMP-2.
       01 LS-SQUARE-D      USAGE COMP-2.
       01 LS-TERM-D        USAGE COMP-2.
       01 LS-COEF-D        USAGE COMP-2.
       01 LS-FLUX-D        USAGE COMP-2.
       01 LS-FLUX-OUT-D    USAGE COMP-2.
       01 LS-FLUX-IN-D     USAGE COMP-2.

      *> Divergence and update chain.
       01 LS-DELTA-SOUTH   USAGE COMP-1.
       01 LS-DELTA-NORTH   USAGE COMP-1.
       01 LS-DELTA-EAST    USAGE COMP-1.
       01 LS-DELTA-WEST    USAGE COMP-1.
       01 LS-HAVE-NORTH    USAGE BINARY-LONG.
       01 LS-HAVE-WEST     USAGE BINARY-LONG.
       01 LS-HAVE-IN       USAGE BINARY-LONG.
       01 LS-CURRENT-F     USAGE COMP-1.
       01 LS-CURRENT-D     USAGE COMP-2.
       01 LS-INCOMING-F    USAGE COMP-1.
       01 LS-DIVERGENCE    USAGE COMP-1.
       01 LS-DIV-SOUTH     USAGE COMP-1.
       01 LS-DIV-EAST      USAGE COMP-1.
       01 LS-CHANGE        USAGE COMP-1.
       01 LS-GAMMA-CHANGE  USAGE COMP-1.
       01 LS-VALUE         USAGE COMP-1.
       01 LS-VALUE-D       USAGE COMP-2.
       01 LS-CHANGE-D      USAGE COMP-2.
       01 LS-GAMMA-CHANGE-D USAGE COMP-2.
       01 LS-UPDATE-D      USAGE COMP-2.
       01 LS-GAMMA-F       USAGE COMP-1.
       01 LS-ZERO-F        USAGE COMP-1 VALUE 0.0.
       01 LS-ONE-F         USAGE COMP-1 VALUE 1.0.
       01 LS-ZERO-D        USAGE COMP-2 VALUE 0.0.
       01 LS-ONE-D         USAGE COMP-2 VALUE 1.0.

       LINKAGE SECTION.
      *> Buffers are float32 planes of L-SIZE elements (row-major). The
      *> OCCURS maximum is GnuCOBOL's 2 GiB table cap; the adapter
      *> rejects larger inputs with a clear error before calling in.
       01 L-IN.
          05 L-IN-PIX      USAGE COMP-1 OCCURS 1 TO 536870911
                           DEPENDING ON L-SIZE.
       01 L-SCRATCH.
          05 L-SCRATCH-PIX USAGE COMP-1 OCCURS 1 TO 536870911
                           DEPENDING ON L-SIZE.
       01 L-OUT.
          05 L-OUT-PIX     USAGE COMP-1 OCCURS 1 TO 536870911
                           DEPENDING ON L-SIZE.
       01 L-ROWS           USAGE BINARY-LONG.
       01 L-COLS           USAGE BINARY-LONG.
       01 L-SIZE           USAGE BINARY-LONG.
       01 L-NITER          USAGE BINARY-LONG.
       01 L-OPTION         USAGE BINARY-LONG.
      *> 0 = linear, 1 = PeronaMalik1, 2 = PeronaMalik2.
       01 L-KAPPA          USAGE COMP-2.
       01 L-GAMMA          USAGE COMP-2.
       01 L-STEP-S         USAGE COMP-2.
       01 L-STEP-E         USAGE COMP-2.
       01 L-PREC           USAGE BINARY-LONG.
       01 L-RC             USAGE BINARY-LONG.

       PROCEDURE DIVISION USING L-IN L-SCRATCH L-OUT
                                L-ROWS L-COLS L-SIZE
                                L-NITER L-OPTION
                                L-KAPPA L-GAMMA L-STEP-S L-STEP-E
                                L-PREC L-RC.
       MAIN-ENTRY.
           MOVE 0 TO L-RC
           PERFORM DECODE-PRECISION
           IF L-SIZE > 0 AND L-NITER > 0
               PERFORM INITIALIZE-SCRATCH
               PERFORM VARYING LS-IT FROM 1 BY 1 UNTIL LS-IT > L-NITER
                   PERFORM RUN-ITERATION
                   PERFORM REFRESH-SCRATCH
               END-PERFORM
           ELSE
               PERFORM VARYING LS-IDX FROM 1 BY 1 UNTIL LS-IDX > L-SIZE
                   MOVE L-IN-PIX(LS-IDX) TO L-OUT-PIX(LS-IDX)
               END-PERFORM
           END-IF
           GOBACK.

      *> ----------------------------------------------------------------
      *> IEEE-754 primitive wrappers. Operands in WS-A(2)/WS-B(2),
      *> in WS-R(2). COMPUTE runs only when both operands are finite;
      *> special cases reproduce IEEE semantics exactly (the decimal
      *> runtime cannot).
       F32-ADD.
           IF WS-A = WS-A AND WS-B = WS-B
               IF WS-A > WS-FLT-MAX
                   IF WS-B < WS-NEG-MAX
                       MOVE WS-NAN TO WS-R
                   ELSE
                       MOVE WS-PINF TO WS-R
                   END-IF
               ELSE
                   IF WS-A < WS-NEG-MAX
                       IF WS-B > WS-FLT-MAX
                           MOVE WS-NAN TO WS-R
                       ELSE
                           MOVE WS-NINF TO WS-R
                       END-IF
                   ELSE
                       IF WS-B > WS-FLT-MAX
                           MOVE WS-PINF TO WS-R
                       ELSE
                           IF WS-B < WS-NEG-MAX
                               MOVE WS-NINF TO WS-R
                           ELSE
                               COMPUTE WS-R = WS-A + WS-B
                           END-IF
                       END-IF
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN TO WS-R
           END-IF
           .

       F32-SUB.
           IF WS-A = WS-A AND WS-B = WS-B
               IF WS-A > WS-FLT-MAX
                   IF WS-B > WS-FLT-MAX
                       MOVE WS-NAN TO WS-R
                   ELSE
                       MOVE WS-PINF TO WS-R
                   END-IF
               ELSE
                   IF WS-A < WS-NEG-MAX
                       IF WS-B < WS-NEG-MAX
                           MOVE WS-NAN TO WS-R
                       ELSE
                           MOVE WS-NINF TO WS-R
                       END-IF
                   ELSE
                       IF WS-B > WS-FLT-MAX
                           MOVE WS-NINF TO WS-R
                       ELSE
                           IF WS-B < WS-NEG-MAX
                               MOVE WS-PINF TO WS-R
                           ELSE
                               COMPUTE WS-R = WS-A - WS-B
                           END-IF
                       END-IF
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN TO WS-R
           END-IF
           .

      *> Sign of a product is the XOR of operand signs; zeros only
      *> combine with infinities to make NaN, else COMPUTE is exact.
       F32-MUL.
           IF WS-A = WS-A AND WS-B = WS-B
               IF WS-A = LS-ZERO-F AND (WS-B > WS-FLT-MAX
                               OR WS-B < WS-NEG-MAX)
                   MOVE WS-NAN TO WS-R
               ELSE
                   IF WS-B = LS-ZERO-F AND (WS-A > WS-FLT-MAX
                                   OR WS-A < WS-NEG-MAX)
                       MOVE WS-NAN TO WS-R
                   ELSE
                       IF WS-A > WS-FLT-MAX OR WS-A < WS-NEG-MAX
                           PERFORM F32-MUL-INF
                       ELSE
                           IF WS-B > WS-FLT-MAX OR WS-B < WS-NEG-MAX
                               PERFORM F32-MUL-INF
                           ELSE
                               COMPUTE WS-R = WS-A * WS-B
                           END-IF
                       END-IF
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN TO WS-R
           END-IF
           .

       F32-MUL-INF.
      *> Sign tests use data-item zeros: comparisons against numeric
      *> literals decimalize the float operand, so (+inf > 0) is false.
           IF (WS-A > LS-ZERO-F AND WS-B > LS-ZERO-F) OR
           (WS-A < LS-ZERO-F AND WS-B < LS-ZERO-F)
               MOVE WS-PINF TO WS-R
           ELSE
               MOVE WS-NINF TO WS-R
           END-IF
           .

      *> IEEE division: x/inf -> zero, x/0 -> signed infinity, 0/0 and
      *> inf/inf -> NaN. The parity matrix only produces +0 divisors.
       F32-DIV.
           IF WS-A = WS-A AND WS-B = WS-B
               IF WS-B > WS-FLT-MAX OR WS-B < WS-NEG-MAX
                   MOVE LS-ZERO-F TO WS-R
               ELSE
                   IF WS-B = LS-ZERO-F
                       IF WS-A = LS-ZERO-F
                           MOVE WS-NAN TO WS-R
                       ELSE
                           IF WS-A > LS-ZERO-F
                               MOVE WS-PINF TO WS-R
                           ELSE
                               MOVE WS-NINF TO WS-R
                           END-IF
                       END-IF
                   ELSE
                       IF WS-A > WS-FLT-MAX
                           IF WS-B > LS-ZERO-F
                               MOVE WS-PINF TO WS-R
                           ELSE
                               MOVE WS-NINF TO WS-R
                           END-IF
                       ELSE
                           IF WS-A < WS-NEG-MAX
                               IF WS-B > LS-ZERO-F
                                   MOVE WS-NINF TO WS-R
                               ELSE
                                   MOVE WS-PINF TO WS-R
                               END-IF
                           ELSE
                               COMPUTE WS-R = WS-A / WS-B
                           END-IF
                       END-IF
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN TO WS-R
           END-IF
           .

      *> exp(WS-A): exp(-inf) = 0, exp(+inf) = +inf, NaN stays NaN;
      *> finite arguments run through the C exponential (matches NumPy's
      *> expf within one ulp, covered by the documented tolerance).
       F32-EXP.
           IF WS-A = WS-A
               IF WS-A > WS-FLT-MAX
                   MOVE WS-PINF TO WS-R
               ELSE
                   IF WS-A < WS-NEG-MAX
                       MOVE LS-ZERO-F TO WS-R
                   ELSE
                       COMPUTE WS-R = FUNCTION EXP(WS-A)
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN TO WS-R
           END-IF
           .

       F64-ADD.
           IF WS-A2 = WS-A2 AND WS-B2 = WS-B2
               IF WS-A2 > WS-DBL-MAX
                   IF WS-B2 < WS-NEG-DBL-MAX
                       MOVE WS-NAN2 TO WS-R2
                   ELSE
                       MOVE WS-PINF2 TO WS-R2
                   END-IF
               ELSE
                   IF WS-A2 < WS-NEG-DBL-MAX
                       IF WS-B2 > WS-DBL-MAX
                           MOVE WS-NAN2 TO WS-R2
                       ELSE
                           MOVE WS-NINF2 TO WS-R2
                       END-IF
                   ELSE
                       IF WS-B2 > WS-DBL-MAX
                           MOVE WS-PINF2 TO WS-R2
                       ELSE
                           IF WS-B2 < WS-NEG-DBL-MAX
                               MOVE WS-NINF2 TO WS-R2
                           ELSE
                               COMPUTE WS-R2 = WS-A2 + WS-B2
                           END-IF
                       END-IF
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN2 TO WS-R2
           END-IF
           .

       F64-SUB.
           IF WS-A2 = WS-A2 AND WS-B2 = WS-B2
               IF WS-A2 > WS-DBL-MAX
                   IF WS-B2 > WS-DBL-MAX
                       MOVE WS-NAN2 TO WS-R2
                   ELSE
                       MOVE WS-PINF2 TO WS-R2
                   END-IF
               ELSE
                   IF WS-A2 < WS-NEG-DBL-MAX
                       IF WS-B2 < WS-NEG-DBL-MAX
                           MOVE WS-NAN2 TO WS-R2
                       ELSE
                           MOVE WS-NINF2 TO WS-R2
                       END-IF
                   ELSE
                       IF WS-B2 > WS-DBL-MAX
                           MOVE WS-NINF2 TO WS-R2
                       ELSE
                           IF WS-B2 < WS-NEG-DBL-MAX
                               MOVE WS-PINF2 TO WS-R2
                           ELSE
                               COMPUTE WS-R2 = WS-A2 - WS-B2
                           END-IF
                       END-IF
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN2 TO WS-R2
           END-IF
           .

       F64-MUL.
           IF WS-A2 = WS-A2 AND WS-B2 = WS-B2
               IF WS-A2 = 0 AND (WS-B2 > WS-DBL-MAX
                                OR WS-B2 < WS-NEG-DBL-MAX)
                   MOVE WS-NAN2 TO WS-R2
               ELSE
                   IF WS-B2 = 0 AND (WS-A2 > WS-DBL-MAX
                                    OR WS-A2 < WS-NEG-DBL-MAX)
                       MOVE WS-NAN2 TO WS-R2
                   ELSE
                       IF WS-A2 > WS-DBL-MAX OR WS-A2 < WS-NEG-DBL-MAX
                           PERFORM F64-MUL-INF
                       ELSE
                           IF WS-B2 > WS-DBL-MAX
                           OR WS-B2 < WS-NEG-DBL-MAX
                               PERFORM F64-MUL-INF
                           ELSE
                               COMPUTE WS-R2 = WS-A2 * WS-B2
                           END-IF
                       END-IF
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN2 TO WS-R2
           END-IF
           .

       F64-MUL-INF.
      *> Data-item zeros again: literal comparisons decimalize +inf to 0.
           IF (WS-A2 > LS-ZERO-D AND WS-B2 > LS-ZERO-D) OR
           (WS-A2 < LS-ZERO-D AND WS-B2 < LS-ZERO-D)
               MOVE WS-PINF2 TO WS-R2
           ELSE
               MOVE WS-NINF2 TO WS-R2
           END-IF
           .

       F64-DIV.
           IF WS-A2 = WS-A2 AND WS-B2 = WS-B2
               IF WS-B2 > WS-DBL-MAX OR WS-B2 < WS-NEG-DBL-MAX
                   MOVE LS-ZERO-D TO WS-R2
               ELSE
                   IF WS-B2 = LS-ZERO-D
                       IF WS-A2 = LS-ZERO-D
                           MOVE WS-NAN2 TO WS-R2
                       ELSE
                           IF WS-A2 > LS-ZERO-D
                               MOVE WS-PINF2 TO WS-R2
                           ELSE
                               MOVE WS-NINF2 TO WS-R2
                           END-IF
                       END-IF
                   ELSE
                       IF WS-A2 > WS-DBL-MAX
                           IF WS-B2 > LS-ZERO-D
                               MOVE WS-PINF2 TO WS-R2
                           ELSE
                               MOVE WS-NINF2 TO WS-R2
                           END-IF
                       ELSE
                           IF WS-A2 < WS-NEG-DBL-MAX
                               IF WS-B2 > LS-ZERO-D
                                   MOVE WS-NINF2 TO WS-R2
                               ELSE
                                   MOVE WS-PINF2 TO WS-R2
                               END-IF
                           ELSE
                               COMPUTE WS-R2 = WS-A2 / WS-B2
                           END-IF
                       END-IF
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN2 TO WS-R2
           END-IF
           .

      *> exp(WS-A2) with the same IEEE special cases; finite arguments
      *> run through the C exponential (libm exp), matching NumPy.
       F64-EXP.
           IF WS-A2 = WS-A2
               IF WS-A2 > WS-DBL-MAX
                   MOVE WS-PINF2 TO WS-R2
               ELSE
                   IF WS-A2 < WS-NEG-DBL-MAX
                       MOVE LS-ZERO-D TO WS-R2
                   ELSE
                       COMPUTE WS-R2 = FUNCTION EXP(WS-A2)
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN2 TO WS-R2
           END-IF
           .

      *> Exact widening f32 -> f64 (numeric MOVEs go through decimal and
      *> destroy non-finite payloads; finite widening is exact).
       F64-OF-F32.
           IF WS-A = WS-A
               IF WS-A > WS-FLT-MAX
                   MOVE WS-PINF2 TO WS-R2
               ELSE
                   IF WS-A < WS-NEG-MAX
                       MOVE WS-NINF2 TO WS-R2
                   ELSE
                       COMPUTE WS-R2 = WS-A
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN2 TO WS-R2
           END-IF
           .

      *> Exact narrowing f64 -> f32: NaN and infinities map 1:1, values
      *> beyond the f32 cast threshold overflow to infinity (NumPy
      *> semantics), finite values round to nearest via COMPUTE.
       F32-OF-F64.
           IF WS-A2 = WS-A2
               IF WS-A2 > WS-F32-OVER
                   MOVE WS-PINF TO WS-R
               ELSE
                   IF WS-A2 < WS-NEG-F32-OVER
                       MOVE WS-NINF TO WS-R
                   ELSE
                       COMPUTE WS-R = WS-A2
                   END-IF
               END-IF
           ELSE
               MOVE WS-NAN TO WS-R
           END-IF
           .

      *> ----------------------------------------------------------------
       DECODE-PRECISION.
      *> Decode the NumPy scalar-promotion bitmask once per call.
           COMPUTE LS-KAPPA-DOUBLE = FUNCTION MOD(L-PREC, 2)
           COMPUTE LS-PREC-REST = L-PREC / 2
           COMPUTE LS-SOUTH-DOUBLE = FUNCTION MOD(LS-PREC-REST, 2)
           COMPUTE LS-PREC-REST = LS-PREC-REST / 2
           COMPUTE LS-EAST-DOUBLE = FUNCTION MOD(LS-PREC-REST, 2)
           COMPUTE LS-PREC-REST = LS-PREC-REST / 2
           COMPUTE LS-GAMMA-DOUBLE = FUNCTION MOD(LS-PREC-REST, 2)
           .

      *> Seed the scratch plane from the (never written) input buffer.
       INITIALIZE-SCRATCH.
           PERFORM VARYING LS-IDX FROM 1 BY 1 UNTIL LS-IDX > L-SIZE
               MOVE L-IN-PIX(LS-IDX) TO L-SCRATCH-PIX(LS-IDX)
           END-PERFORM
           .

      *> Jacobi sweep: deltas read the scratch plane, updates land in
      *> output plane; the reference refreshes imgout once per sweep.
       RUN-ITERATION.
           PERFORM VARYING LS-I FROM 1 BY 1 UNTIL LS-I > L-ROWS
               PERFORM VARYING LS-J FROM 1 BY 1 UNTIL LS-J > L-COLS
                   COMPUTE LS-IDX = (LS-I - 1) * L-COLS + LS-J
                   MOVE L-SCRATCH-PIX(LS-IDX) TO LS-VALUE
                   PERFORM COMPUTE-DELTAS
      *> South axis: S = flux(south delta) minus flux(north delta).
                   PERFORM SET-AXIS-SOUTH
                   MOVE LS-DELTA-SOUTH TO LS-FLUX-DELTA
                   PERFORM CALC-FLUX
                   MOVE LS-FLUX-D TO LS-FLUX-OUT-D
                   MOVE LS-HAVE-NORTH TO LS-HAVE-IN
                   IF LS-HAVE-IN = 1
                       MOVE LS-DELTA-NORTH TO LS-FLUX-DELTA
                       PERFORM CALC-FLUX
                       MOVE LS-FLUX-D TO LS-FLUX-IN-D
                   END-IF
                   PERFORM DIVERGE
                   MOVE LS-DIVERGENCE TO LS-DIV-SOUTH
      *> East axis: E = flux(east delta) minus flux(west delta).
                   PERFORM SET-AXIS-EAST
                   MOVE LS-DELTA-EAST TO LS-FLUX-DELTA
                   PERFORM CALC-FLUX
                   MOVE LS-FLUX-D TO LS-FLUX-OUT-D
                   MOVE LS-HAVE-WEST TO LS-HAVE-IN
                   IF LS-HAVE-IN = 1
                       MOVE LS-DELTA-WEST TO LS-FLUX-DELTA
                       PERFORM CALC-FLUX
                       MOVE LS-FLUX-D TO LS-FLUX-IN-D
                   END-IF
                   PERFORM DIVERGE
                   MOVE LS-DIVERGENCE TO LS-DIV-EAST
                   PERFORM DIRECTIONAL-SUM
                   PERFORM APPLY-UPDATE
               END-PERFORM
           END-PERFORM
           .

      *> Publish the swept plane back to the read side for next pass.
       REFRESH-SCRATCH.
           PERFORM VARYING LS-IDX FROM 1 BY 1 UNTIL LS-IDX > L-SIZE
               MOVE L-OUT-PIX(LS-IDX) TO L-SCRATCH-PIX(LS-IDX)
           END-PERFORM
           .

      *> Directional deltas with the reference's boundary sentinels:
      *> zero delta beyond bottom/right edges (still flux-evaluated),
      *> no incoming flux at the top/left edges.
       COMPUTE-DELTAS.
           IF LS-I < L-ROWS
               MOVE L-SCRATCH-PIX(LS-IDX + L-COLS) TO WS-A
               MOVE LS-VALUE TO WS-B
               PERFORM F32-SUB
               MOVE WS-R TO LS-DELTA-SOUTH
           ELSE
               MOVE LS-ZERO-F TO LS-DELTA-SOUTH
           END-IF
           IF LS-I > 1
               MOVE LS-VALUE TO WS-A
               MOVE L-SCRATCH-PIX(LS-IDX - L-COLS) TO WS-B
               PERFORM F32-SUB
               MOVE WS-R TO LS-DELTA-NORTH
               MOVE 1 TO LS-HAVE-NORTH
           ELSE
               MOVE 0 TO LS-HAVE-NORTH
           END-IF
           IF LS-J < L-COLS
               MOVE L-SCRATCH-PIX(LS-IDX + 1) TO WS-A
               MOVE LS-VALUE TO WS-B
               PERFORM F32-SUB
               MOVE WS-R TO LS-DELTA-EAST
           ELSE
               MOVE LS-ZERO-F TO LS-DELTA-EAST
           END-IF
           IF LS-J > 1
               MOVE LS-VALUE TO WS-A
               MOVE L-SCRATCH-PIX(LS-IDX - 1) TO WS-B
               PERFORM F32-SUB
               MOVE WS-R TO LS-DELTA-WEST
               MOVE 1 TO LS-HAVE-WEST
           ELSE
               MOVE 0 TO LS-HAVE-WEST
           END-IF
           .

      *> South axis parameters for CALC-FLUX. Linear flux ignores both
      *> step and the promotion flags (the coefficient is exactly one).
       SET-AXIS-SOUTH.
      *> Same-type MOVE is bit-preserving; COMPUTE would decimalize.
           MOVE L-STEP-S TO LS-A-STEP-D
           MOVE LS-SOUTH-DOUBLE TO LS-A-STEP-DOUBLE
           PERFORM SET-AXIS-DOUBLE
           .

       SET-AXIS-EAST.
           MOVE L-STEP-E TO LS-A-STEP-D
           MOVE LS-EAST-DOUBLE TO LS-A-STEP-DOUBLE
           PERFORM SET-AXIS-DOUBLE
           .

       SET-AXIS-DOUBLE.
           IF L-OPTION = 0
               MOVE 0 TO LS-A-DOUBLE
           ELSE
               IF LS-KAPPA-DOUBLE = 1 OR LS-A-STEP-DOUBLE = 1
                   MOVE 1 TO LS-A-DOUBLE
               ELSE
                   MOVE 0 TO LS-A-DOUBLE
               END-IF
           END-IF
           .

      *> Flux of LS-FLUX-DELTA through the axis in LS-A-*. The
      *> coefficient runs in the precision NumPy used; the flux result
      *> float64 because the Rust flux() returns f64. DIVERGE narrows it
      *> back exactly where the reference stores into float32 arrays.
       CALC-FLUX.
           IF L-OPTION = 0
               MOVE LS-FLUX-DELTA TO WS-A
               PERFORM F64-OF-F32
               MOVE WS-R2 TO LS-FLUX-D
           ELSE
               IF LS-KAPPA-DOUBLE = 1
                   MOVE LS-FLUX-DELTA TO WS-A
                   PERFORM F64-OF-F32
                   MOVE WS-R2 TO WS-A2
                   MOVE L-KAPPA TO WS-B2
                   PERFORM F64-DIV
                   MOVE WS-R2 TO LS-RATIO-D
                   MOVE LS-RATIO-D TO WS-A2
                   MOVE LS-RATIO-D TO WS-B2
                   PERFORM F64-MUL
                   MOVE WS-R2 TO LS-SQUARE-D
                   IF L-OPTION = 1
                       MOVE LS-ZERO-D TO WS-A2
                       MOVE LS-SQUARE-D TO WS-B2
                       PERFORM F64-SUB
                       MOVE WS-R2 TO WS-A2
                       PERFORM F64-EXP
                       MOVE WS-R2 TO LS-COEF-D
                   ELSE
                       MOVE LS-ONE-D TO WS-A2
                       MOVE LS-SQUARE-D TO WS-B2
                       PERFORM F64-ADD
                       MOVE WS-R2 TO LS-TERM-D
                       MOVE LS-ONE-D TO WS-A2
                       MOVE LS-TERM-D TO WS-B2
                       PERFORM F64-DIV
                       MOVE WS-R2 TO LS-COEF-D
                   END-IF
               ELSE
                   MOVE L-KAPPA TO WS-A2
                   PERFORM F32-OF-F64
                   MOVE WS-R TO LS-KAPPA-F
                   MOVE LS-FLUX-DELTA TO WS-A
                   MOVE LS-KAPPA-F TO WS-B
                   PERFORM F32-DIV
                   MOVE WS-R TO LS-RATIO-F
                   MOVE LS-RATIO-F TO WS-A
                   MOVE LS-RATIO-F TO WS-B
                   PERFORM F32-MUL
                   MOVE WS-R TO LS-SQUARE-F
                   IF L-OPTION = 1
                       MOVE LS-ZERO-F TO WS-A
                       MOVE LS-SQUARE-F TO WS-B
                       PERFORM F32-SUB
                       MOVE WS-R TO WS-A
                       PERFORM F32-EXP
                       MOVE WS-R TO LS-COEF-F
                   ELSE
                       MOVE LS-ONE-F TO WS-A
                       MOVE LS-SQUARE-F TO WS-B
                       PERFORM F32-ADD
                       MOVE WS-R TO LS-TERM-F
                       MOVE LS-ONE-F TO WS-A
                       MOVE LS-TERM-F TO WS-B
                       PERFORM F32-DIV
                       MOVE WS-R TO LS-COEF-F
                   END-IF
                   MOVE LS-COEF-F TO WS-A
                   PERFORM F64-OF-F32
                   MOVE WS-R2 TO LS-COEF-D
               END-IF
               IF LS-A-DOUBLE = 1
                   MOVE LS-COEF-D TO WS-A2
                   MOVE LS-A-STEP-D TO WS-B2
                   PERFORM F64-DIV
                   MOVE WS-R2 TO LS-COEF-D
                   MOVE LS-COEF-D TO WS-A2
                   MOVE LS-FLUX-DELTA TO WS-A
                   PERFORM F64-OF-F32
                   MOVE WS-R2 TO WS-B2
                   PERFORM F64-MUL
                   MOVE WS-R2 TO LS-FLUX-D
               ELSE
                   MOVE LS-A-STEP-D TO WS-A2
                   PERFORM F32-OF-F64
                   MOVE WS-R TO LS-STEP-F
                   MOVE LS-COEF-F TO WS-A
                   MOVE LS-STEP-F TO WS-B
                   PERFORM F32-DIV
                   MOVE WS-R TO LS-COEF-F
                   MOVE LS-COEF-F TO WS-A
                   MOVE LS-FLUX-DELTA TO WS-B
                   PERFORM F32-MUL
                   MOVE WS-R TO LS-FLUX-F
                   MOVE LS-FLUX-F TO WS-A
                   PERFORM F64-OF-F32
                   MOVE WS-R2 TO LS-FLUX-D
               END-IF
           END-IF
           .

      *> NS/EW first receive the outgoing flux narrowed to float32, then
      *> subtract the incoming flux in the axis's promoted precision
      *> (rust/src/diffusion.rs divergence()).
       DIVERGE.
           MOVE LS-FLUX-OUT-D TO WS-A2
           PERFORM F32-OF-F64
           MOVE WS-R TO LS-CURRENT-F
           IF LS-HAVE-IN = 0
               MOVE LS-CURRENT-F TO LS-DIVERGENCE
           ELSE
               IF LS-A-DOUBLE = 1
                   MOVE LS-CURRENT-F TO WS-A
                   PERFORM F64-OF-F32
                   MOVE WS-R2 TO LS-CURRENT-D
                   MOVE LS-CURRENT-D TO WS-A2
                   MOVE LS-FLUX-IN-D TO WS-B2
                   PERFORM F64-SUB
                   MOVE WS-R2 TO WS-A2
                   PERFORM F32-OF-F64
                   MOVE WS-R TO LS-DIVERGENCE
               ELSE
                   MOVE LS-FLUX-IN-D TO WS-A2
                   PERFORM F32-OF-F64
                   MOVE WS-R TO LS-INCOMING-F
                   MOVE LS-CURRENT-F TO WS-A
                   MOVE LS-INCOMING-F TO WS-B
                   PERFORM F32-SUB
                   MOVE WS-R TO LS-DIVERGENCE
               END-IF
           END-IF
           .

      *> Reference NaN masking: zero each direction's NaN independently,
      *> add, then restore NaN only where both directions were NaN.
      *> Self-equality is false exactly for NaN; the addition itself can
      *> still see opposing infinities, so it runs through F32-ADD.
       DIRECTIONAL-SUM.
           IF LS-DIV-SOUTH = LS-DIV-SOUTH
               IF LS-DIV-EAST = LS-DIV-EAST
                   MOVE LS-DIV-SOUTH TO WS-A
                   MOVE LS-DIV-EAST TO WS-B
                   PERFORM F32-ADD
                   MOVE WS-R TO LS-CHANGE
               ELSE
                   MOVE LS-DIV-SOUTH TO LS-CHANGE
               END-IF
           ELSE
               MOVE LS-DIV-EAST TO LS-CHANGE
           END-IF
           .

       APPLY-UPDATE.
           IF LS-GAMMA-DOUBLE = 1
               MOVE LS-VALUE TO WS-A
               PERFORM F64-OF-F32
               MOVE WS-R2 TO LS-VALUE-D
               MOVE LS-CHANGE TO WS-A
               PERFORM F64-OF-F32
               MOVE WS-R2 TO LS-CHANGE-D
               MOVE L-GAMMA TO WS-A2
               MOVE LS-CHANGE-D TO WS-B2
               PERFORM F64-MUL
               MOVE WS-R2 TO LS-GAMMA-CHANGE-D
               MOVE LS-VALUE-D TO WS-A2
               MOVE LS-GAMMA-CHANGE-D TO WS-B2
               PERFORM F64-ADD
               MOVE WS-R2 TO LS-UPDATE-D
               MOVE LS-UPDATE-D TO WS-A2
               PERFORM F32-OF-F64
               MOVE WS-R TO L-OUT-PIX(LS-IDX)
           ELSE
               MOVE L-GAMMA TO WS-A2
               PERFORM F32-OF-F64
               MOVE WS-R TO LS-GAMMA-F
               MOVE LS-GAMMA-F TO WS-A
               MOVE LS-CHANGE TO WS-B
               PERFORM F32-MUL
               MOVE WS-R TO LS-GAMMA-CHANGE
               MOVE LS-VALUE TO WS-A
               MOVE LS-GAMMA-CHANGE TO WS-B
               PERFORM F32-ADD
               MOVE WS-R TO L-OUT-PIX(LS-IDX)
           END-IF
           .
