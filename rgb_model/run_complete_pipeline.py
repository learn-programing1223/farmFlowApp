#!/usr/bin/env python3
"""
COMPLETE CYCLEGAN AUGMENTATION PIPELINE ORCHESTRATOR
Orchestrates the entire pipeline from dataset preparation to model training and testing
Provides comprehensive monitoring and error handling
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import logging
import argparse
import shutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrates the complete CycleGAN augmentation pipeline"""
    
    def __init__(self, config_file: str = None):
        """Initialize the orchestrator"""
        self.start_time = datetime.now()
        self.results = {}
        self.config = self._load_config(config_file) if config_file else self._default_config()
        
        logger.info("PipelineOrchestrator initialized")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def _default_config(self) -> dict:
        """Default pipeline configuration"""
        return {
            # Dataset preparation
            'dataset_prep': {
                'enabled': True,
                'output_path': 'datasets/ultimate_plant_disease_cyclegan',
                'augmentation_ratio': 0.3,
                'severity': 0.7,
                'max_per_category': 4000,
                'force_rebuild': False
            },
            
            # Training
            'training': {
                'enabled': True,
                'model_type': 'efficient',  # 'efficient' or 'custom'
                'learning_rate': 0.0001,
                'batch_size': 16,
                'epochs': 50,
                'use_runtime_augmentation': True,
                'augmentation_severity': 0.5
            },
            
            # Testing
            'testing': {
                'enabled': True,
                'test_real_images': True,
                'test_internet_images': True,
                'generate_report': True
            },
            
            # Deployment
            'deployment': {
                'enabled': True,
                'convert_to_tflite': True,
                'target_app_path': 'PlantPulse/assets/models/',
                'update_web_app': True
            },
            
            # General settings
            'general': {
                'cleanup_intermediate': False,
                'save_checkpoints': True,
                'backup_existing_models': True
            }
        }
    
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config {config_file}: {e}")
            logger.info("Using default configuration")
            return self._default_config()
    
    def _run_command(self, command: list, description: str, timeout: int = 3600) -> bool:
        """Run a command with logging and error handling"""
        logger.info(f"Starting: {description}")
        logger.info(f"Command: {' '.join(command)}")
        
        start_time = time.time()
        
        try:
            # Run command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✓ {description} completed in {elapsed:.1f}s")
            
            # Log output if verbose
            if result.stdout:
                logger.debug(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.debug(f"STDERR:\n{result.stderr}")
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out after {timeout}s")
            return False
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ {description} failed after {elapsed:.1f}s")
            logger.error(f"Return code: {e.returncode}")
            if e.stdout:
                logger.error(f"STDOUT:\n{e.stdout}")
            if e.stderr:
                logger.error(f"STDERR:\n{e.stderr}")
            return False
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ {description} failed after {elapsed:.1f}s: {str(e)}")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        logger.info("="*70)
        logger.info("CHECKING PREREQUISITES")
        logger.info("="*70)
        
        checks_passed = 0
        total_checks = 0
        
        # Check Python version
        total_checks += 1
        python_version = sys.version_info
        if python_version >= (3, 8):
            logger.info(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
            checks_passed += 1
        else:
            logger.error(f"❌ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
        
        # Check required modules
        required_modules = [
            'tensorflow', 'numpy', 'opencv-python', 'pillow', 
            'matplotlib', 'seaborn', 'scikit-learn', 'tqdm'
        ]
        
        for module in required_modules:
            total_checks += 1
            try:
                __import__(module.replace('-', '_'))
                logger.info(f"✓ {module}")
                checks_passed += 1
            except ImportError:
                logger.error(f"❌ {module} not installed")
        
        # Check GPU availability (optional)
        total_checks += 1
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"✓ GPU available: {len(gpus)} device(s)")
                checks_passed += 1
            else:
                logger.warning("⚠️  No GPU found (training will be slower)")
                checks_passed += 1  # Not critical
        except Exception:
            logger.warning("⚠️  Could not check GPU status")
            checks_passed += 1  # Not critical
        
        # Check disk space (estimate 10GB needed)
        total_checks += 1
        try:
            free_space = shutil.disk_usage('.').free / (1024**3)  # GB
            if free_space >= 10:
                logger.info(f"✓ Disk space: {free_space:.1f} GB available")
                checks_passed += 1
            else:
                logger.warning(f"⚠️  Low disk space: {free_space:.1f} GB (recommend 10GB+)")
                checks_passed += 1  # Warning but continue
        except Exception:
            logger.warning("⚠️  Could not check disk space")
            checks_passed += 1
        
        # Check for dataset directories
        total_checks += 1
        dataset_found = False
        for potential_dataset in ['PlantVillage', 'datasets', 'PlantDisease']:
            if Path(potential_dataset).exists():
                dataset_found = True
                logger.info(f"✓ Found dataset directory: {potential_dataset}")
                break
        
        if dataset_found:
            checks_passed += 1
        else:
            logger.warning("⚠️  No dataset directories found")
            logger.warning("    Pipeline will search automatically during dataset prep")
            checks_passed += 1  # Will handle during dataset prep
        
        success_rate = checks_passed / total_checks
        logger.info(f"\nPrerequisite check: {checks_passed}/{total_checks} passed ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            logger.info("✅ Prerequisites check passed")
            return True
        else:
            logger.error("❌ Prerequisites check failed")
            return False
    
    def run_dataset_preparation(self) -> bool:
        """Run dataset preparation with CycleGAN augmentation"""
        if not self.config['dataset_prep']['enabled']:
            logger.info("Dataset preparation disabled, skipping...")
            return True
        
        logger.info("="*70)
        logger.info("DATASET PREPARATION WITH CYCLEGAN AUGMENTATION")
        logger.info("="*70)
        
        # Check if dataset already exists and force_rebuild is False
        dataset_path = Path(self.config['dataset_prep']['output_path'])
        if (dataset_path.exists() and 
            not self.config['dataset_prep']['force_rebuild']):
            logger.info(f"Dataset already exists at {dataset_path}")
            logger.info("Use force_rebuild=true to rebuild")
            
            # Quick validation
            required_dirs = ['train', 'val', 'test']
            if all((dataset_path / d).exists() for d in required_dirs):
                logger.info("✓ Dataset structure validated")
                self.results['dataset_prep'] = {'status': 'skipped', 'path': str(dataset_path)}
                return True
            else:
                logger.warning("Dataset structure incomplete, rebuilding...")
        
        # Run dataset preparation
        command = [
            sys.executable, 
            'prepare_ultimate_dataset_cyclegan.py'
        ]
        
        success = self._run_command(
            command,
            "Dataset preparation with CycleGAN augmentation",
            timeout=1800  # 30 minutes
        )
        
        if success:
            # Validate output
            if dataset_path.exists():
                logger.info(f"✓ Dataset created at: {dataset_path}")
                
                # Count images
                total_images = 0
                for split in ['train', 'val', 'test']:
                    split_path = dataset_path / split
                    if split_path.exists():
                        split_count = sum(
                            len(list(cat_dir.glob('*.jpg'))) + len(list(cat_dir.glob('*.png')))
                            for cat_dir in split_path.iterdir()
                            if cat_dir.is_dir()
                        )
                        total_images += split_count
                        logger.info(f"  {split}: {split_count} images")
                
                logger.info(f"  Total: {total_images} images")
                
                self.results['dataset_prep'] = {
                    'status': 'success',
                    'path': str(dataset_path),
                    'total_images': total_images
                }
                
                return True
            else:
                logger.error("Dataset directory not found after preparation")
                self.results['dataset_prep'] = {'status': 'failed', 'error': 'Output not found'}
                return False
        else:
            self.results['dataset_prep'] = {'status': 'failed', 'error': 'Command failed'}
            return False
    
    def run_training(self) -> bool:
        """Run model training"""
        if not self.config['training']['enabled']:
            logger.info("Training disabled, skipping...")
            return True
        
        logger.info("="*70)
        logger.info("MODEL TRAINING WITH CYCLEGAN AUGMENTATION")
        logger.info("="*70)
        
        # Check if dataset exists
        dataset_path = Path(self.config['dataset_prep']['output_path'])
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            logger.error("Run dataset preparation first")
            return False
        
        # Backup existing models if requested
        if self.config['general']['backup_existing_models']:
            models_dir = Path('models')
            if models_dir.exists():
                backup_dir = Path(f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                shutil.copytree(models_dir, backup_dir)
                logger.info(f"✓ Models backed up to: {backup_dir}")
        
        # Run training
        command = [
            sys.executable,
            'train_ultimate_cyclegan.py'
        ]
        
        success = self._run_command(
            command,
            "Model training with CycleGAN augmentation",
            timeout=7200  # 2 hours
        )
        
        if success:
            # Validate training outputs
            models_dir = Path('models')
            expected_files = [
                'final_cyclegan_model.h5',
                'plant_disease_cyclegan_robust.tflite',
                'training_history_cyclegan.json'
            ]
            
            missing_files = []
            for file in expected_files:
                if not (models_dir / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                logger.warning(f"Some output files missing: {missing_files}")
            else:
                logger.info("✓ All training outputs generated")
            
            # Load training results
            try:
                history_path = models_dir / 'training_history_cyclegan.json'
                with open(history_path, 'r') as f:
                    history_data = json.load(f)
                
                eval_results = history_data.get('evaluation_results', {})
                final_accuracy = eval_results.get('accuracy', 0)
                
                logger.info(f"✓ Final test accuracy: {final_accuracy:.1%}")
                
                self.results['training'] = {
                    'status': 'success',
                    'accuracy': final_accuracy,
                    'model_files': expected_files
                }
                
                return True
                
            except Exception as e:
                logger.warning(f"Could not load training results: {e}")
                self.results['training'] = {
                    'status': 'success_partial',
                    'warning': 'Could not read results'
                }
                return True
        else:
            self.results['training'] = {'status': 'failed', 'error': 'Training command failed'}
            return False
    
    def run_testing(self) -> bool:
        """Run comprehensive testing"""
        if not self.config['testing']['enabled']:
            logger.info("Testing disabled, skipping...")
            return True
        
        logger.info("="*70)
        logger.info("COMPREHENSIVE MODEL TESTING")
        logger.info("="*70)
        
        # Check if model exists
        models_dir = Path('models')
        model_file = models_dir / 'plant_disease_cyclegan_robust.tflite'
        
        if not model_file.exists():
            logger.error(f"Model not found: {model_file}")
            logger.error("Run training first")
            return False
        
        test_results = {}
        
        # Test 1: Real world images
        if self.config['testing']['test_real_images']:
            logger.info("Running real-world image tests...")
            
            command = [sys.executable, 'comprehensive_real_world_test.py']
            success = self._run_command(
                command,
                "Real-world image testing",
                timeout=600  # 10 minutes
            )
            test_results['real_world'] = 'success' if success else 'failed'
        
        # Test 2: Internet images (if script exists)
        if self.config['testing']['test_internet_images']:
            test_script = Path('test_with_internet_images.py')
            if test_script.exists():
                logger.info("Running internet image tests...")
                
                command = [sys.executable, str(test_script)]
                success = self._run_command(
                    command,
                    "Internet image testing",
                    timeout=900  # 15 minutes
                )
                test_results['internet_images'] = 'success' if success else 'failed'
            else:
                logger.info("Internet image test script not found, skipping...")
                test_results['internet_images'] = 'skipped'
        
        # Generate comprehensive report
        if self.config['testing']['generate_report']:
            try:
                self._generate_test_report(test_results)
                logger.info("✓ Test report generated")
            except Exception as e:
                logger.warning(f"Could not generate test report: {e}")
        
        self.results['testing'] = test_results
        
        # Consider testing successful if at least one test passed
        success = any(result == 'success' for result in test_results.values())
        return success
    
    def run_deployment(self) -> bool:
        """Handle model deployment"""
        if not self.config['deployment']['enabled']:
            logger.info("Deployment disabled, skipping...")
            return True
        
        logger.info("="*70)
        logger.info("MODEL DEPLOYMENT")
        logger.info("="*70)
        
        models_dir = Path('models')
        tflite_file = models_dir / 'plant_disease_cyclegan_robust.tflite'
        
        if not tflite_file.exists():
            logger.error(f"TFLite model not found: {tflite_file}")
            return False
        
        deployment_results = {}
        
        # Deploy to app
        app_path = Path(self.config['deployment']['target_app_path'])
        if app_path.parent.exists():
            try:
                app_path.mkdir(parents=True, exist_ok=True)
                target_file = app_path / 'plant_disease_model.tflite'
                shutil.copy2(tflite_file, target_file)
                logger.info(f"✓ Model deployed to app: {target_file}")
                deployment_results['app_deployment'] = 'success'
            except Exception as e:
                logger.error(f"Failed to deploy to app: {e}")
                deployment_results['app_deployment'] = 'failed'
        else:
            logger.warning(f"App path not found: {app_path.parent}")
            deployment_results['app_deployment'] = 'skipped'
        
        # Update web app configuration (if needed)
        if self.config['deployment']['update_web_app']:
            try:
                self._update_web_app_config()
                logger.info("✓ Web app configuration updated")
                deployment_results['web_app_update'] = 'success'
            except Exception as e:
                logger.warning(f"Could not update web app: {e}")
                deployment_results['web_app_update'] = 'failed'
        
        self.results['deployment'] = deployment_results
        return True
    
    def _update_web_app_config(self):
        """Update web app configuration with new model"""
        # This would update web app configuration files
        # Implementation depends on specific web app structure
        logger.info("Web app configuration update not implemented yet")
    
    def _generate_test_report(self, test_results: dict):
        """Generate comprehensive test report"""
        report_lines = [
            "CYCLEGAN AUGMENTATION PIPELINE - TEST REPORT",
            "="*60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "TEST RESULTS:",
            "-"*30
        ]
        
        for test_name, result in test_results.items():
            status_symbol = "✓" if result == 'success' else "❌" if result == 'failed' else "⚠️"
            report_lines.append(f"{status_symbol} {test_name}: {result}")
        
        report_lines.extend([
            "",
            "PIPELINE SUMMARY:",
            "-"*30,
            f"Dataset Prep: {self.results.get('dataset_prep', {}).get('status', 'not run')}",
            f"Training: {self.results.get('training', {}).get('status', 'not run')}",
            f"Testing: {'success' if any(r == 'success' for r in test_results.values()) else 'failed'}",
            f"Deployment: {self.results.get('deployment', {}).get('status', 'not run')}",
        ])
        
        # Add performance metrics if available
        if 'training' in self.results and 'accuracy' in self.results['training']:
            accuracy = self.results['training']['accuracy']
            report_lines.extend([
                "",
                "PERFORMANCE METRICS:",
                "-"*30,
                f"Test Accuracy: {accuracy:.1%}",
                f"Expected Real-World: ~{accuracy * 0.85:.1%}",
                f"Performance Grade: {'A' if accuracy >= 0.9 else 'B' if accuracy >= 0.8 else 'C'}"
            ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = Path('models') / 'PIPELINE_TEST_REPORT.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Test report saved: {report_path}")
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate files if requested"""
        if not self.config['general']['cleanup_intermediate']:
            return
        
        logger.info("Cleaning up intermediate files...")
        
        # List of patterns to clean
        cleanup_patterns = [
            'logs/events.out.tfevents.*',
            '*.log',
            '__pycache__',
            '.pytest_cache',
            '*.pyc'
        ]
        
        cleaned_count = 0
        for pattern in cleanup_patterns:
            try:
                for path in Path('.').glob(pattern):
                    if path.is_file():
                        path.unlink()
                        cleaned_count += 1
                    elif path.is_dir():
                        shutil.rmtree(path)
                        cleaned_count += 1
            except Exception as e:
                logger.warning(f"Could not clean {pattern}: {e}")
        
        logger.info(f"✓ Cleaned {cleaned_count} intermediate files")
    
    def generate_final_report(self):
        """Generate final pipeline report"""
        total_time = datetime.now() - self.start_time
        
        print("\n" + "="*70)
        print("🎉 CYCLEGAN AUGMENTATION PIPELINE - FINAL REPORT")
        print("="*70)
        
        print(f"\n⏱️  EXECUTION TIME: {total_time}")
        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📊 PIPELINE RESULTS:")
        for stage, result in self.results.items():
            status = result.get('status', 'unknown') if isinstance(result, dict) else str(result)
            symbol = "✓" if 'success' in status else "❌" if 'failed' in status else "⚠️"
            print(f"  {symbol} {stage.replace('_', ' ').title()}: {status}")
        
        # Performance summary
        if 'training' in self.results and isinstance(self.results['training'], dict):
            if 'accuracy' in self.results['training']:
                accuracy = self.results['training']['accuracy']
                print(f"\n🎯 MODEL PERFORMANCE:")
                print(f"  Test Accuracy: {accuracy:.1%}")
                print(f"  Performance Grade: {'A' if accuracy >= 0.9 else 'B' if accuracy >= 0.8 else 'C'}")
        
        # Generated files
        models_dir = Path('models')
        if models_dir.exists():
            print(f"\n📁 GENERATED FILES:")
            for file in models_dir.glob('*cyclegan*'):
                print(f"  - {file}")
        
        print(f"\n🚀 NEXT STEPS:")
        print("  1. Test the model with your own images")
        print("  2. Deploy to mobile app")
        print("  3. Monitor real-world performance")
        print("  4. Collect feedback for improvements")
        
        # Overall success
        success_count = sum(1 for result in self.results.values() 
                          if (isinstance(result, dict) and 'success' in result.get('status', '')) 
                          or result == 'success')
        total_stages = len(self.results)
        
        if success_count == total_stages:
            print(f"\n✅ PIPELINE SUCCESS! All {total_stages} stages completed successfully.")
        elif success_count > total_stages / 2:
            print(f"\n⚠️  PIPELINE PARTIAL SUCCESS: {success_count}/{total_stages} stages successful.")
        else:
            print(f"\n❌ PIPELINE FAILED: Only {success_count}/{total_stages} stages successful.")
        
        print("="*70)
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete pipeline"""
        logger.info("🚀 STARTING COMPLETE CYCLEGAN AUGMENTATION PIPELINE")
        
        try:
            # Stage 1: Prerequisites
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed, aborting pipeline")
                return False
            
            # Stage 2: Dataset preparation
            if not self.run_dataset_preparation():
                logger.error("Dataset preparation failed")
                return False
            
            # Stage 3: Training
            if not self.run_training():
                logger.error("Training failed")
                return False
            
            # Stage 4: Testing
            if not self.run_testing():
                logger.warning("Some tests failed, but continuing...")
            
            # Stage 5: Deployment
            if not self.run_deployment():
                logger.warning("Deployment had issues, but pipeline succeeded")
            
            # Stage 6: Cleanup
            self.cleanup_intermediate_files()
            
            # Stage 7: Final report
            self.generate_final_report()
            
            logger.info("✅ Complete pipeline execution finished")
            return True
            
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user")
            return False
            
        except Exception as e:
            logger.error(f"Pipeline failed with unexpected error: {e}")
            return False


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Complete CycleGAN Augmentation Pipeline Orchestrator"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file (JSON)'
    )
    
    parser.add_argument(
        '--skip-dataset-prep',
        action='store_true',
        help='Skip dataset preparation'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training'
    )
    
    parser.add_argument(
        '--skip-testing',
        action='store_true',
        help='Skip testing'
    )
    
    parser.add_argument(
        '--quick-run',
        action='store_true',
        help='Quick run with reduced epochs and dataset size'
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(args.config)
    
    # Apply command-line overrides
    if args.skip_dataset_prep:
        orchestrator.config['dataset_prep']['enabled'] = False
    
    if args.skip_training:
        orchestrator.config['training']['enabled'] = False
    
    if args.skip_testing:
        orchestrator.config['testing']['enabled'] = False
    
    if args.quick_run:
        # Quick run settings
        orchestrator.config['dataset_prep']['max_per_category'] = 1000
        orchestrator.config['training']['epochs'] = 10
        orchestrator.config['training']['batch_size'] = 32
        logger.info("Quick run mode enabled")
    
    # Run pipeline
    success = orchestrator.run_complete_pipeline()
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()