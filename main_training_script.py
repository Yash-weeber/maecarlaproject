def main():
    """Main training script with theoretical analysis and validation"""

    print("=" * 80)
    print("MULTI-ROBOT AUTONOMOUS DRIVING SYSTEM")
    print("Complete Training Pipeline with Theoretical Analysis")
    print("=" * 80)

    # Step 1: Generate theoretical analysis
    print("\n[STEP 1/4] Generating Theoretical Analysis...")
    analyzer = TheoreticalAnalysis(
        conflict_radius=30.0,
        min_safe_distance=5.0,
        max_velocity=15.0,
        max_deceleration=8.0,
        reaction_time=1.0
    )

    proof1 = analyzer.prove_collision_avoidance()
    proof2 = analyzer.prove_deadlock_freedom(max_vehicles=20)

    analyzer.generate_latex_report("theoretical_proofs.tex")
    analyzer.generate_markdown_report("theoretical_analysis.md")
    analyzer.visualize_safety_zones("validation_plots")

    print("✓ Theoretical analysis complete")

    # Step 2: Setup training configuration
    print("\n[STEP 2/4] Setting up training configuration...")
    config = {
        'num_vehicles': 5,
        'town': 'Town03',
        'carla_port': 2000,
        'max_episode_steps': 1000,
        'learning_rate': 3e-4,
        'n_epochs': 10
    }

    print(f"Configuration: {config}")

    # Step 3: Train model
    print("\n[STEP 3/4] Training model...")
    trainer = StableMultiAgentTrainer(config)
    trainer.setup_training()
    trainer.setup_evaluation()

    # FIX: Train for specified timesteps (can handle 1B+ now)
    model = trainer.train_agents(total_timesteps=1000000)  # Start with 1M for testing

    print("✓ Training complete")

    # Step 4: Final validation
    print("\n[STEP 4/4] Final validation...")
    val1 = trainer.validation_metrics.validate_collision_avoidance()
    val2 = trainer.validation_metrics.validate_deadlock_freedom()

    print("\nVALIDATION RESULTS:")
    print(f"  Property 1 (Collision Avoidance): {'✓ SATISFIED' if val1['property_satisfied'] else '✗ VIOLATED'}")
    print(f"  Property 2 (Deadlock Freedom): {'✓ SATISFIED' if val2['property_satisfied'] else '✗ VIOLATED'}")

    print("\n" + "=" * 80)
    print("ALL STEPS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  ✓ theoretical_proofs.tex")
    print("  ✓ theoretical_analysis.md")
    print("  ✓ validation_plots/collision_avoidance_zones.png")
    print("  ✓ validation_plots/deadlock_freedom_proof.png")
    print("  ✓ validation_results/collision_avoidance_validation.png")
    print("  ✓ validation_results/deadlock_freedom_validation.png")
    print("  ✓ validation_results/training_progress.png")
    print("  ✓ validation_results/validation_report.txt")
    print("  ✓ validation_results/metrics_data.csv")
    print("  ✓ experiments/models/final_model_*/")


if __name__ == "__main__":
    main()
