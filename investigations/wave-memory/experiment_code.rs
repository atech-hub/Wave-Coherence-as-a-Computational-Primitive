// ─── Memory test command ──────────────────────────────────────

fn print_memory_test_help() {
    println!("kerr-engine memory-test — Test wave memory injection");
    println!();
    println!("USAGE:");
    println!("    kerr-engine memory-test <checkpoint> <data> [OPTIONS]");
    println!();
    println!("ARGUMENTS:");
    println!("    checkpoint    Path to trained checkpoint (.bin)");
    println!("    data          Path to training data (for vocabulary)");
    println!();
    println!("OPTIONS:");
    println!("    --word        Use word-level tokenizer");
    println!("    --bpe FILE    Use BPE tokenizer");
    println!("    --seed N      RNG seed (default: 42)");
    println!("    --tokens N    Tokens to generate per test (default: 200)");
    println!();
    println!("Tests wave memory injection at alpha values from 0.0 to 1.0.");
    println!("Generates text with each alpha and measures perplexity change.");
    println!("Alpha=0.0 is the baseline (no memory = bit-identical to normal).");
    println!();
    println!("EXAMPLE:");
    println!("    kerr-engine memory-test checkpoint_final.bin data/input.txt");
}

fn run_memory_test(args: &[String]) {
    println!("Wave Memory Injection Test\n");

    // Parse args
    let checkpoint_path = args.get(2).unwrap_or_else(|| {
        eprintln!("Usage: kerr-engine memory-test <checkpoint> <data>");
        std::process::exit(1);
    });
    let data_path = args.get(3).unwrap_or_else(|| {
        eprintln!("Usage: kerr-engine memory-test <checkpoint> <data>");
        std::process::exit(1);
    });
    let word_level = args.iter().any(|a| a == "--word");
    let bpe_path: Option<String> = args.iter()
        .position(|a| a == "--bpe")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string());
    let seed: u64 = args.iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let n_tokens: usize = args.iter()
        .position(|a| a == "--tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    // Load checkpoint
    println!("Loading checkpoint: {checkpoint_path}");
    let state = checkpoint::load(checkpoint_path).expect("Failed to load checkpoint");
    let model = state.model;
    let n_bands = model.config.n_bands;
    let n_ode_layers = model.config.n_layers - 1; // Block 0 is PerBandLinear, rest are Kerr-ODE
    println!("  Model: {} layers, {} bands, {} vocab",
        model.config.n_layers, n_bands, model.vocab_size);

    // Load dataset for vocab/decode
    println!("Loading data: {data_path}");
    let dataset = if let Some(ref bpe) = bpe_path {
        data::Dataset::from_file_bpe(data_path, bpe, 0.9)
    } else if word_level {
        data::Dataset::from_file_words(data_path, 0.9, 3)
    } else {
        data::Dataset::from_file(data_path)
    };

    // Create synthetic memory: deterministic random values
    println!("\nGenerating synthetic memory (seed={seed}, {n_ode_layers} ODE layers, {n_bands} bands)");
    let base_r: Vec<Vec<f32>> = (0..n_ode_layers).map(|layer| {
        let mut state = seed.wrapping_mul(layer as u64 + 1).wrapping_add(12345);
        (0..n_bands).map(|_| {
            state = xorshift_mem(state);
            (state & 0xFFFFFF) as f32 / 0xFFFFFF as f32 - 0.5
        }).collect()
    }).collect();
    let base_s: Vec<Vec<f32>> = (0..n_ode_layers).map(|layer| {
        let mut state = seed.wrapping_mul(layer as u64 + 1).wrapping_add(67890);
        (0..n_bands).map(|_| {
            state = xorshift_mem(state);
            (state & 0xFFFFFF) as f32 / 0xFFFFFF as f32 - 0.5
        }).collect()
    }).collect();

    // Test at various alpha values
    let alphas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
    println!("\n{:>8} {:>10} {:>10}  Sample (first 80 chars)", "Alpha", "Perplexity", "Delta%");
    println!("{}", "-".repeat(90));

    let mut baseline_ppl = 0.0f32;

    for &alpha in &alphas {
        let mut gen_rng = rng::Rng::new(seed);

        // Scale memory by alpha
        let scaled_r: Vec<Vec<f32>> = base_r.iter()
            .map(|r| r.iter().map(|&v| v * alpha).collect())
            .collect();
        let scaled_s: Vec<Vec<f32>> = base_s.iter()
            .map(|s| s.iter().map(|&v| v * alpha).collect())
            .collect();

        // Build offset slices
        let offsets: Vec<(&[f32], &[f32])> = scaled_r.iter().zip(scaled_s.iter())
            .map(|(r, s)| (r.as_slice(), s.as_slice()))
            .collect();

        let memory = if alpha == 0.0 { None } else { Some(offsets.as_slice()) };

        // Generate tokens
        let start_idx = *dataset.token_to_idx.get("\n").unwrap_or(
            dataset.token_to_idx.values().next().unwrap_or(&0)
        );
        let mut tokens = vec![start_idx];

        for _ in 0..n_tokens {
            let block_size = model.config.block_size;
            let start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
            let context = &tokens[start..];

            let logits_all = model.forward_with_memory(context, memory);
            let logits = logits_all.last().unwrap();

            // Sample with temperature
            let temp = 0.8f32;
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: Vec<f32> = logits.iter().map(|&l| ((l - max_l) / temp).exp()).collect();
            let sum_exp: f32 = exp_l.iter().sum();
            let probs: Vec<f32> = exp_l.iter().map(|e| e / sum_exp).collect();

            let mut r = gen_rng.next_f32();
            let mut chosen = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                r -= p;
                if r <= 0.0 { chosen = i; break; }
            }

            tokens.push(chosen);
        }

        // Compute perplexity on generated sequence
        let mut total_log_prob = 0.0f64;
        for i in 1..tokens.len() {
            let context = &tokens[..i];
            let start = if context.len() > model.config.block_size {
                context.len() - model.config.block_size
            } else { 0 };
            let logits_all = model.forward_with_memory(&context[start..], memory);
            let logits = logits_all.last().unwrap();

            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum_exp: f32 = exp_l.iter().sum();
            let target = tokens[i];
            if target < exp_l.len() {
                total_log_prob += (exp_l[target] / sum_exp).ln() as f64;
            }
        }
        let ppl = (-total_log_prob / (tokens.len() - 1) as f64).exp() as f32;

        if alpha == 0.0 { baseline_ppl = ppl; }
        let delta = if baseline_ppl > 0.0 {
            (ppl - baseline_ppl) / baseline_ppl * 100.0
        } else { 0.0 };

        let text = dataset.decode(&tokens[1..]);
        let preview: String = text.chars().take(80).collect();
        let preview = preview.replace('\n', " ");

        println!("{:>8.3} {:>10.2} {:>+10.1}%  {}", alpha, ppl, delta, preview);
    }

    println!("\n--- Interpretation ---");
    println!("Alpha=0.0 is the baseline (no memory, identical to normal forward pass).");
    println!("Small alpha (0.001-0.01): output should differ slightly, perplexity similar.");
    println!("Medium alpha (0.05-0.1): noticeable difference, perplexity may increase.");
    println!("Large alpha (0.5-1.0): significant perturbation, perplexity should degrade.");
    println!("If perplexity barely changes at ANY alpha, initial conditions don't matter.");
}

fn xorshift_mem(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

// ─── Memory accumulation test ─────────────────────────────────

fn print_memory_accumulate_help() {
    println!("kerr-engine memory-accumulate — Test memory accumulation stability");
    println!();
    println!("USAGE:");
    println!("    kerr-engine memory-accumulate <checkpoint> <data> [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --convos N     Number of conversations (default: 10)");
    println!("    --tokens N     Tokens per conversation (default: 200)");
    println!("    --alpha F      Memory injection strength (default: 0.05)");
    println!("    --decay F      Within-conversation EMA decay (default: 0.99)");
    println!("    --beta F       Cross-conversation merge rate (default: 0.95)");
    println!("    --seed N       RNG seed (default: 42)");
    println!("    --word         Use word-level tokenizer");
    println!("    --bpe FILE     Use BPE tokenizer");
    println!();
    println!("Runs N conversations, accumulating wave memory between them.");
    println!("Reports per-conversation energy, convergence, and harmonic census.");
    println!("Tests whether the EMA accumulator converges or diverges.");
    println!();
    println!("EXAMPLE:");
    println!("    kerr-engine memory-accumulate checkpoint_iter3000.bin data/input.txt");
}

fn run_memory_accumulate_test(args: &[String]) {
    println!("Wave Memory Accumulation Test (Experiment 2)\n");

    // Parse args
    let checkpoint_path = args.get(2).unwrap_or_else(|| {
        eprintln!("Usage: kerr-engine memory-accumulate <checkpoint> <data>");
        std::process::exit(1);
    });
    let data_path = args.get(3).unwrap_or_else(|| {
        eprintln!("Usage: kerr-engine memory-accumulate <checkpoint> <data>");
        std::process::exit(1);
    });
    let n_convos: usize = parse_flag_usize(args, "--convos").unwrap_or(10);
    let n_tokens: usize = parse_flag_usize(args, "--tokens").unwrap_or(200);
    let alpha: f32 = parse_flag_f32_val(args, "--alpha").unwrap_or(0.05);
    let decay: f32 = parse_flag_f32_val(args, "--decay").unwrap_or(0.99);
    let beta: f32 = parse_flag_f32_val(args, "--beta").unwrap_or(0.95);
    let seed: u64 = parse_flag_usize(args, "--seed").unwrap_or(42) as u64;
    let word_level = args.iter().any(|a| a == "--word");
    let bpe_path: Option<String> = args.iter()
        .position(|a| a == "--bpe")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string());

    // Load model
    println!("Loading checkpoint: {checkpoint_path}");
    let state = checkpoint::load(checkpoint_path).expect("Failed to load checkpoint");
    let model = state.model;
    let n_bands = model.config.n_bands;
    let n_ode_layers = model.config.n_layers - 1;
    println!("  Model: {} layers, {} bands, {} vocab", model.config.n_layers, n_bands, model.vocab_size);

    // Load dataset
    println!("Loading data: {data_path}");
    let dataset = if let Some(ref bpe) = bpe_path {
        data::Dataset::from_file_bpe(data_path, bpe, 0.9)
    } else if word_level {
        data::Dataset::from_file_words(data_path, 0.9, 3)
    } else {
        data::Dataset::from_file(data_path)
    };

    println!("\nConfig: alpha={alpha}, decay={decay}, beta={beta}");
    println!("Running {n_convos} conversations x {n_tokens} tokens each\n");

    // Memory state: per-layer (r, s) vectors
    let mut memory_r: Vec<Vec<f32>> = (0..n_ode_layers).map(|_| vec![0.0f32; n_bands]).collect();
    let mut memory_s: Vec<Vec<f32>> = (0..n_ode_layers).map(|_| vec![0.0f32; n_bands]).collect();

    // Track energy per conversation for convergence
    let mut energy_history: Vec<Vec<f32>> = Vec::new(); // [convo][layer]
    let mut prev_energy: Vec<f32> = vec![0.0; n_ode_layers];

    println!("{:>6} {:>12} {:>12} {:>12} {:>12}  Sample (40 chars)",
        "Convo", "Total E", "Delta E%", "Max Band E", "Active");
    println!("{}", "-".repeat(90));

    for convo in 0..n_convos {
        let mut gen_rng = rng::Rng::new(seed + convo as u64 * 7919);

        // Build memory offsets (scaled by alpha)
        let scaled_r: Vec<Vec<f32>> = memory_r.iter()
            .map(|r| r.iter().map(|&v| v * alpha).collect()).collect();
        let scaled_s: Vec<Vec<f32>> = memory_s.iter()
            .map(|s| s.iter().map(|&v| v * alpha).collect()).collect();
        let offsets: Vec<(&[f32], &[f32])> = scaled_r.iter().zip(scaled_s.iter())
            .map(|(r, s)| (r.as_slice(), s.as_slice())).collect();
        let memory = if convo == 0 { None } else { Some(offsets.as_slice()) };

        // Generate conversation
        let start_idx = *dataset.token_to_idx.get("\n").unwrap_or(
            dataset.token_to_idx.values().next().unwrap_or(&0));
        let mut tokens = vec![start_idx];

        for _ in 0..n_tokens {
            let block_size = model.config.block_size;
            let start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
            let logits_all = model.forward_with_memory(&tokens[start..], memory);
            let logits = logits_all.last().unwrap();

            let temp = 0.8f32;
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: Vec<f32> = logits.iter().map(|&l| ((l - max_l) / temp).exp()).collect();
            let sum_exp: f32 = exp_l.iter().sum();
            let probs: Vec<f32> = exp_l.iter().map(|e| e / sum_exp).collect();

            let mut r = gen_rng.next_f32();
            let mut chosen = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                r -= p;
                if r <= 0.0 { chosen = i; break; }
            }
            tokens.push(chosen);
        }

        // Extract ODE states (last block_size tokens only)
        let ext_start = if tokens.len() > model.config.block_size {
            tokens.len() - model.config.block_size
        } else { 0 };
        let ode_states = model.extract_ode_states(&tokens[ext_start..], memory);

        // EMA accumulate: within-conversation average already done in extract_ode_states
        // Now merge into persistent memory with beta
        let mut layer_energies = Vec::new();
        for (layer, (r_avg, s_avg)) in ode_states.iter().enumerate() {
            for k in 0..n_bands {
                memory_r[layer][k] = beta * memory_r[layer][k] + (1.0 - beta) * r_avg[k];
                memory_s[layer][k] = beta * memory_s[layer][k] + (1.0 - beta) * s_avg[k];
            }
            let energy: f32 = (0..n_bands)
                .map(|k| memory_r[layer][k] * memory_r[layer][k]
                       + memory_s[layer][k] * memory_s[layer][k])
                .sum();
            layer_energies.push(energy);
        }

        let total_energy: f32 = layer_energies.iter().sum();
        let delta_pct = if prev_energy.iter().sum::<f32>() > 0.0 {
            let prev_total: f32 = prev_energy.iter().sum();
            (total_energy - prev_total) / prev_total * 100.0
        } else { 0.0 };

        // Max band energy across all layers
        let mut max_band_e = 0.0f32;
        for l in 0..n_ode_layers {
            for k in 0..n_bands {
                let e = memory_r[l][k] * memory_r[l][k] + memory_s[l][k] * memory_s[l][k];
                if e > max_band_e { max_band_e = e; }
            }
        }

        // Active bands (>1% of max)
        let threshold = max_band_e * 0.01;
        let mut active = 0usize;
        for l in 0..n_ode_layers {
            for k in 0..n_bands {
                let e = memory_r[l][k] * memory_r[l][k] + memory_s[l][k] * memory_s[l][k];
                if e > threshold { active += 1; }
            }
        }

        let text = dataset.decode(&tokens[1..]);
        let preview: String = text.chars().take(40).collect();
        let preview = preview.replace('\n', " ");

        println!("{:>6} {:>12.6} {:>+12.1}% {:>12.6} {:>5}/{:<5}  {}",
            convo, total_energy, delta_pct, max_band_e,
            active, n_ode_layers * n_bands, preview);

        prev_energy = layer_energies.clone();
        energy_history.push(layer_energies);
    }

    // Convergence analysis
    println!("\n--- Convergence Analysis ---\n");

    // Per-layer energy progression
    println!("Per-layer energy across conversations:");
    println!("{:>6}  {}", "Convo", (0..n_ode_layers).map(|l| format!("Layer {l:>4}")).collect::<Vec<_>>().join("  "));
    for (c, energies) in energy_history.iter().enumerate() {
        let cols: String = energies.iter().map(|e| format!("{e:>10.6}")).collect::<Vec<_>>().join("  ");
        println!("{:>6}  {}", c, cols);
    }

    // Energy delta between consecutive conversations
    println!("\nEnergy delta (%) between consecutive conversations:");
    for c in 1..energy_history.len() {
        let deltas: String = energy_history[c].iter().zip(energy_history[c-1].iter())
            .map(|(curr, prev)| {
                if *prev > 0.0 { format!("{:>+10.2}%", (curr - prev) / prev * 100.0) }
                else { format!("{:>10}", "n/a") }
            })
            .collect::<Vec<_>>().join("  ");
        println!("{:>6}  {}", c, deltas);
    }

    // Top 5 bands by energy in final memory
    println!("\nTop 10 bands in final memory (across all layers):");
    let mut all_bands: Vec<(usize, usize, f32)> = Vec::new();
    for l in 0..n_ode_layers {
        for k in 0..n_bands {
            let e = memory_r[l][k] * memory_r[l][k] + memory_s[l][k] * memory_s[l][k];
            all_bands.push((l, k, e));
        }
    }
    all_bands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!("  {:>6} {:>6} {:>12}", "Layer", "Band", "Energy");
    for (l, k, e) in all_bands.iter().take(10) {
        println!("  {:>6} {:>6} {:>12.6}", l, k, e);
    }

    // Final verdict
    let early_energy: f32 = energy_history.first().map(|e| e.iter().sum()).unwrap_or(0.0);
    let late_energy: f32 = energy_history.last().map(|e| e.iter().sum()).unwrap_or(0.0);
    let final_delta = if early_energy > 0.0 {
        ((late_energy - early_energy) / early_energy * 100.0).abs()
    } else { 0.0 };

    // Check last 3 conversations for stability
    let stable = if energy_history.len() >= 4 {
        let last3: Vec<f32> = energy_history[energy_history.len()-3..]
            .iter().map(|e| e.iter().sum::<f32>()).collect();
        let mean: f32 = last3.iter().sum::<f32>() / 3.0;
        let max_dev = last3.iter().map(|&e| ((e - mean) / mean * 100.0).abs())
            .fold(0.0f32, f32::max);
        max_dev < 10.0 // Less than 10% deviation = stable
    } else { false };

    println!("\n--- Verdict ---");
    println!("First conversation energy: {early_energy:.6}");
    println!("Last conversation energy:  {late_energy:.6}");
    println!("Total drift: {final_delta:.1}%");
    println!("Last 3 conversations stable (< 10% deviation): {}", if stable { "YES" } else { "NO" });

    if stable {
        println!("\nRESULT: STABLE. Memory accumulation converges.");
        println!("The EMA + implicit regularisation (damping) prevents divergence.");
    } else if late_energy > early_energy * 10.0 {
        println!("\nRESULT: DIVERGENT. Energy growing without bound.");
        println!("Try lower decay ({decay}) or higher beta ({beta}).");
    } else {
        println!("\nRESULT: INCONCLUSIVE. Run more conversations or adjust parameters.");
    }
}

fn parse_flag_usize(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_flag_f32_val(args: &[String], flag: &str) -> Option<f32> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

// ─── Semantic memory test ─────────────────────────────────────

fn print_memory_semantic_help() {
    println!("kerr-engine memory-semantic — Test semantic content in wave memory");
    println!();
    println!("USAGE:");
    println!("    kerr-engine memory-semantic <checkpoint> <data> [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --convos N     Conversations per topic (default: 5)");
    println!("    --tokens N     Tokens per conversation (default: 200)");
    println!("    --gen N        Tokens to generate for comparison (default: 300)");
    println!("    --alpha F      Injection strength (default: 0.05)");
    println!("    --seed N       RNG seed (default: 42)");
    println!("    --word         Use word-level tokenizer");
    println!("    --bpe FILE     Use BPE tokenizer");
    println!();
    println!("Accumulates separate memories from love-themed and war-themed passages,");
    println!("then generates from a neutral prompt with each. Compares output and");
    println!("harmonic census to test whether memory carries semantic content.");
}

fn run_memory_semantic_test(args: &[String]) {
    println!("Wave Memory Semantic Test (Experiment 3)\n");

    let checkpoint_path = args.get(2).unwrap_or_else(|| {
        eprintln!("Usage: kerr-engine memory-semantic <checkpoint> <data>");
        std::process::exit(1);
    });
    let data_path = args.get(3).unwrap_or_else(|| {
        eprintln!("Usage: kerr-engine memory-semantic <checkpoint> <data>");
        std::process::exit(1);
    });
    let n_convos: usize = parse_flag_usize(args, "--convos").unwrap_or(5);
    let n_tokens: usize = parse_flag_usize(args, "--tokens").unwrap_or(200);
    let n_gen: usize = parse_flag_usize(args, "--gen").unwrap_or(300);
    let alpha: f32 = parse_flag_f32_val(args, "--alpha").unwrap_or(0.05);
    let beta: f32 = 0.99; // from Experiment 2
    let seed: u64 = parse_flag_usize(args, "--seed").unwrap_or(42) as u64;
    let word_level = args.iter().any(|a| a == "--word");
    let bpe_path: Option<String> = args.iter()
        .position(|a| a == "--bpe")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string());

    // Load model
    println!("Loading checkpoint: {checkpoint_path}");
    let state = checkpoint::load(checkpoint_path).expect("Failed to load checkpoint");
    let model = state.model;
    let n_bands = model.config.n_bands;
    let n_ode_layers = model.config.n_layers - 1;
    println!("  Model: {} layers, {} bands", model.config.n_layers, n_bands);

    // Load dataset
    println!("Loading data: {data_path}");
    let dataset = if let Some(ref bpe) = bpe_path {
        data::Dataset::from_file_bpe(data_path, bpe, 0.9)
    } else if word_level {
        data::Dataset::from_file_words(data_path, 0.9, 3)
    } else {
        data::Dataset::from_file(data_path)
    };

    // Shakespeare love and war prompts (seeds for conversation topics)
    let love_prompts = [
        "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate.\n",
        "But soft, what light through yonder window breaks?\nIt is the east, and Juliet is the sun.\n",
        "Love looks not with the eyes, but with the mind,\nAnd therefore is winged Cupid painted blind.\n",
        "My bounty is as boundless as the sea,\nMy love as deep; the more I give to thee,\n",
        "Hear my soul speak:\nThe very instant that I saw you, did my heart fly to your service.\n",
        "She loved me for the dangers I had passed,\nAnd I loved her that she did pity them.\n",
        "Doubt thou the stars are fire,\nDoubt that the sun doth move,\nDoubt truth to be a liar,\nBut never doubt I love.\n",
        "Love all, trust a few, do wrong to none.\n",
    ];

    let war_prompts = [
        "Once more unto the breach, dear friends, once more;\nOr close the wall up with our English dead.\n",
        "Cry 'Havoc!', and let slip the dogs of war.\n",
        "Now is the winter of our discontent\nMade glorious summer by this sun of York.\n",
        "We few, we happy few, we band of brothers;\nFor he today that sheds his blood with me\n",
        "Uneasy lies the head that wears a crown.\n",
        "The evil that men do lives after them;\nThe good is oft interred with their bones.\n",
        "Cowards die many times before their deaths;\nThe valiant never taste of death but once.\n",
        "A horse! a horse! my kingdom for a horse!\n",
    ];

    println!("\nConfig: alpha={alpha}, beta={beta}");
    println!("Phase 1: Accumulating {n_convos} love conversations...\n");

    // Accumulate love memory
    let love_memory = accumulate_topic_memory(
        &model, &dataset, &love_prompts, n_convos, n_tokens,
        alpha, beta, n_bands, n_ode_layers, seed,
    );

    println!("\nPhase 2: Accumulating {n_convos} war conversations...\n");

    // Accumulate war memory
    let war_memory = accumulate_topic_memory(
        &model, &dataset, &war_prompts, n_convos, n_tokens,
        alpha, beta, n_bands, n_ode_layers, seed + 10000,
    );

    // Harmonic census comparison
    println!("\n--- Harmonic Census Comparison ---\n");
    println!("Top 10 bands by energy:\n");
    println!("{:>22}    {:>22}", "LOVE MEMORY", "WAR MEMORY");
    println!("{:>6} {:>6} {:>8}    {:>6} {:>6} {:>8}",
        "Layer", "Band", "Energy", "Layer", "Band", "Energy");

    let mut love_bands: Vec<(usize, usize, f32)> = Vec::new();
    let mut war_bands: Vec<(usize, usize, f32)> = Vec::new();
    for l in 0..n_ode_layers {
        for k in 0..n_bands {
            let le = love_memory.0[l][k] * love_memory.0[l][k] + love_memory.1[l][k] * love_memory.1[l][k];
            let we = war_memory.0[l][k] * war_memory.0[l][k] + war_memory.1[l][k] * war_memory.1[l][k];
            love_bands.push((l, k, le));
            war_bands.push((l, k, we));
        }
    }
    love_bands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    war_bands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    for i in 0..10 {
        let (ll, lk, le) = love_bands[i];
        let (wl, wk, we) = war_bands[i];
        let marker = if lk != wk || ll != wl { " <--" } else { "" };
        println!("{:>6} {:>6} {:>8.4}    {:>6} {:>6} {:>8.4}{marker}",
            ll, lk, le, wl, wk, we);
    }

    // Compute correlation between love and war memory profiles
    let love_energy: Vec<f32> = love_bands.iter().map(|(_, _, e)| *e).collect();
    let war_energy: Vec<f32> = war_bands.iter().map(|(_, _, e)| *e).collect();
    // Need both sorted by same index for correlation
    let mut love_by_idx = vec![0.0f32; n_ode_layers * n_bands];
    let mut war_by_idx = vec![0.0f32; n_ode_layers * n_bands];
    for l in 0..n_ode_layers {
        for k in 0..n_bands {
            let idx = l * n_bands + k;
            love_by_idx[idx] = love_memory.0[l][k] * love_memory.0[l][k] + love_memory.1[l][k] * love_memory.1[l][k];
            war_by_idx[idx] = war_memory.0[l][k] * war_memory.0[l][k] + war_memory.1[l][k] * war_memory.1[l][k];
        }
    }
    let corr = pearson_corr(&love_by_idx, &war_by_idx);
    let love_total: f32 = love_by_idx.iter().sum();
    let war_total: f32 = war_by_idx.iter().sum();

    println!("\nEnergy profile correlation: {corr:.4}");
    println!("  (1.0 = identical profiles, 0.0 = uncorrelated, <0 = anti-correlated)");
    println!("  Love total energy: {love_total:.6}");
    println!("  War total energy:  {war_total:.6}");

    // Generate from neutral prompt with each memory
    println!("\n--- Generation Comparison ---\n");
    let neutral_prompt = "\n"; // just a newline = neutral start

    for (label, mem_r, mem_s) in [
        ("NO MEMORY", &vec![vec![0.0f32; n_bands]; n_ode_layers], &vec![vec![0.0f32; n_bands]; n_ode_layers]),
        ("LOVE MEMORY", &love_memory.0, &love_memory.1),
        ("WAR MEMORY", &war_memory.0, &war_memory.1),
    ] {
        let mut gen_rng = rng::Rng::new(seed + 99999); // same RNG for all 3

        let scaled_r: Vec<Vec<f32>> = mem_r.iter()
            .map(|r| r.iter().map(|&v| v * alpha).collect()).collect();
        let scaled_s: Vec<Vec<f32>> = mem_s.iter()
            .map(|s| s.iter().map(|&v| v * alpha).collect()).collect();
        let offsets: Vec<(&[f32], &[f32])> = scaled_r.iter().zip(scaled_s.iter())
            .map(|(r, s)| (r.as_slice(), s.as_slice())).collect();
        let memory = if label == "NO MEMORY" { None } else { Some(offsets.as_slice()) };

        // Encode prompt
        let start_tokens: Vec<usize> = neutral_prompt.chars()
            .map(|c| *dataset.token_to_idx.get(&c.to_string()).unwrap_or(&0))
            .collect();
        let mut tokens = start_tokens;

        for _ in 0..n_gen {
            let block_size = model.config.block_size;
            let start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
            let logits_all = model.forward_with_memory(&tokens[start..], memory);
            let logits = logits_all.last().unwrap();

            let temp = 0.8f32;
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: Vec<f32> = logits.iter().map(|&l| ((l - max_l) / temp).exp()).collect();
            let sum_exp: f32 = exp_l.iter().sum();
            let probs: Vec<f32> = exp_l.iter().map(|e| e / sum_exp).collect();

            let mut r = gen_rng.next_f32();
            let mut chosen = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                r -= p;
                if r <= 0.0 { chosen = i; break; }
            }
            tokens.push(chosen);
        }

        let text = dataset.decode(&tokens[1..]);
        println!("[{label}]");
        let preview: String = text.chars().take(200).collect();
        println!("{}\n", preview);
    }

    println!("--- Verdict ---");
    if corr < 0.9 {
        println!("RESULT: DIFFERENT. Love and war memories have distinct harmonic profiles.");
        println!("Correlation {corr:.4} < 0.9 — the memory carries topic-specific content.");
    } else {
        println!("RESULT: SIMILAR. Love and war memories have near-identical profiles.");
        println!("Correlation {corr:.4} >= 0.9 — the memory captures texture, not topic.");
    }
}

/// Accumulate memory from topic-seeded conversations.
/// Returns (memory_r, memory_s) per layer.
fn accumulate_topic_memory(
    model: &model::ModelWeights,
    dataset: &data::Dataset,
    prompts: &[&str],
    n_convos: usize,
    n_tokens: usize,
    alpha: f32,
    beta: f32,
    n_bands: usize,
    n_ode_layers: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut memory_r: Vec<Vec<f32>> = (0..n_ode_layers).map(|_| vec![0.0f32; n_bands]).collect();
    let mut memory_s: Vec<Vec<f32>> = (0..n_ode_layers).map(|_| vec![0.0f32; n_bands]).collect();

    for convo in 0..n_convos {
        let mut gen_rng = rng::Rng::new(seed + convo as u64 * 7919);

        // Build memory offsets
        let scaled_r: Vec<Vec<f32>> = memory_r.iter()
            .map(|r| r.iter().map(|&v| v * alpha).collect()).collect();
        let scaled_s: Vec<Vec<f32>> = memory_s.iter()
            .map(|s| s.iter().map(|&v| v * alpha).collect()).collect();
        let offsets: Vec<(&[f32], &[f32])> = scaled_r.iter().zip(scaled_s.iter())
            .map(|(r, s)| (r.as_slice(), s.as_slice())).collect();
        let memory = if convo == 0 { None } else { Some(offsets.as_slice()) };

        // Seed with topic prompt (truncate to fit block_size with room for generation)
        let prompt = prompts[convo % prompts.len()];
        let max_prompt = model.config.block_size / 4; // leave room for generation
        let mut tokens: Vec<usize> = prompt.chars()
            .take(max_prompt)
            .map(|c| *dataset.token_to_idx.get(&c.to_string()).unwrap_or(&0))
            .collect();

        // Generate continuation
        for _ in 0..n_tokens {
            let block_size = model.config.block_size;
            let start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
            let logits_all = model.forward_with_memory(&tokens[start..], memory);
            let logits = logits_all.last().unwrap();

            let temp = 0.8f32;
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: Vec<f32> = logits.iter().map(|&l| ((l - max_l) / temp).exp()).collect();
            let sum_exp: f32 = exp_l.iter().sum();
            let probs: Vec<f32> = exp_l.iter().map(|e| e / sum_exp).collect();

            let mut r = gen_rng.next_f32();
            let mut chosen = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                r -= p;
                if r <= 0.0 { chosen = i; break; }
            }
            tokens.push(chosen);
        }

        // Extract ODE states (last block_size tokens only)
        let block_size = model.config.block_size;
        let ext_start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
        let ode_states = model.extract_ode_states(&tokens[ext_start..], memory);

        // Merge into memory
        for (layer, (r_avg, s_avg)) in ode_states.iter().enumerate() {
            for k in 0..n_bands {
                memory_r[layer][k] = beta * memory_r[layer][k] + (1.0 - beta) * r_avg[k];
                memory_s[layer][k] = beta * memory_s[layer][k] + (1.0 - beta) * s_avg[k];
            }
        }

        let total_e: f32 = (0..n_ode_layers).map(|l|
            (0..n_bands).map(|k| memory_r[l][k] * memory_r[l][k] + memory_s[l][k] * memory_s[l][k]).sum::<f32>()
        ).sum();

        let text = dataset.decode(&tokens[tokens.len().saturating_sub(40)..]);
        let preview = text.replace('\n', " ");
        println!("  Convo {convo}: energy={total_e:.6}  ...{preview}");
    }

    (memory_r, memory_s)
}

fn pearson_corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    if n == 0.0 { return 0.0; }
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-12 { 0.0 } else { cov / denom }
}
