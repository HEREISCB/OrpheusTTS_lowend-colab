#!/usr/bin/env python3
"""
OrpheusTTS Benchmark Script

This script tests the concurrent generation capabilities of OrpheusTTS on an H100 GPU.
It runs multiple concurrent TTS generations and measures performance metrics.
"""

import time
import asyncio
import argparse
import numpy as np
import logging
import os
import wave
import concurrent.futures
from datetime import datetime
from orpheus_tts import OrpheusModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Sample texts of varying lengths for more realistic testing
SAMPLE_TEXTS = [
    "Hello, this is a short test of the Orpheus text to speech system.",
    "The quick brown fox jumps over the lazy dog. This pangram contains all the letters of the English alphabet.",
    "In a world where technology evolves at an unprecedented pace, artificial intelligence stands at the forefront of innovation.",
    "Text to speech technology has advanced significantly in recent years, with neural networks producing increasingly natural-sounding voices.",
    "The H100 GPU represents a significant advancement in computational power, enabling more efficient processing of complex AI workloads."
]

class TTSBenchmark:
    def __init__(self, model_name="canopylabs/orpheus-tts-0.1-finetune-prod", concurrent_tasks=200):
        """Initialize the benchmark with the specified model and concurrency level."""
        self.model_name = model_name
        self.concurrent_tasks = concurrent_tasks
        self.model = None
        self.results = []
        self.output_dir = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model(self):
        """Load the OrpheusTTS model."""
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()
        self.model = OrpheusModel(model_name=self.model_name)
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
    def generate_speech(self, text, voice="tara", task_id=0):
        """Generate speech for a single text input and measure performance."""
        if not text:
            text = np.random.choice(SAMPLE_TEXTS)
            
        start_time = time.time()
        logger.info(f"Task {task_id}: Starting generation for text ({len(text)} chars)")
        
        try:
            # Generate speech
            syn_tokens = self.model.generate_speech(
                prompt=text,
                voice=voice,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                max_tokens=2000
            )
            
            # Process the generated audio
            audio_chunks = []
            for chunk in syn_tokens:
                audio_chunks.append(chunk)
                
            # Calculate audio duration
            total_audio_bytes = sum(len(chunk) for chunk in audio_chunks)
            # Assuming 16-bit mono audio at 24kHz
            audio_duration = total_audio_bytes / (24000 * 2)
            
            # Save audio to file (optional)
            output_filename = os.path.join(self.output_dir, f"output_{task_id}.wav")
            with wave.open(output_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                for chunk in audio_chunks:
                    wf.writeframes(chunk)
            
            # Calculate processing time and real-time factor
            processing_time = time.time() - start_time
            rtf = processing_time / audio_duration if audio_duration > 0 else float('inf')
            
            result = {
                "task_id": task_id,
                "text_length": len(text),
                "audio_duration": audio_duration,
                "processing_time": processing_time,
                "rtf": rtf,
                "success": True
            }
            
            logger.info(f"Task {task_id}: Generated {audio_duration:.2f}s audio in {processing_time:.2f}s (RTF: {rtf:.3f})")
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Task {task_id}: Failed after {processing_time:.2f}s - {str(e)}")
            result = {
                "task_id": task_id,
                "text_length": len(text),
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
            
        return result
    
    async def run_async_benchmark(self):
        """Run the benchmark using asyncio for concurrency."""
        if not self.model:
            self.load_model()
            
        logger.info(f"Starting benchmark with {self.concurrent_tasks} concurrent tasks")
        overall_start_time = time.time()
        
        # Create a thread pool for CPU-bound tasks
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrent_tasks)
        loop = asyncio.get_event_loop()
        
        # Create tasks
        tasks = []
        for i in range(self.concurrent_tasks):
            # Cycle through sample texts
            text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
            
            # Create a task that runs in the thread pool
            task = loop.run_in_executor(
                executor, 
                self.generate_speech,
                text, "tara", i
            )
            tasks.append(task)
            
        # Wait for all tasks to complete
        self.results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        overall_time = time.time() - overall_start_time
        successful_tasks = sum(1 for r in self.results if isinstance(r, dict) and r.get("success", False))
        
        logger.info(f"Benchmark completed in {overall_time:.2f} seconds")
        logger.info(f"Successfully completed {successful_tasks}/{self.concurrent_tasks} tasks")
        
        return self.results
        
    def run_benchmark(self):
        """Run the benchmark synchronously."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.run_async_benchmark())
    
    def analyze_results(self):
        """Analyze benchmark results and print statistics."""
        if not self.results:
            logger.warning("No results to analyze")
            return
            
        # Filter out exceptions
        valid_results = [r for r in self.results if isinstance(r, dict)]
        successful_results = [r for r in valid_results if r.get("success", False)]
        failed_results = [r for r in valid_results if not r.get("success", False)]
        exceptions = [r for r in self.results if isinstance(r, Exception)]
        
        if not successful_results:
            logger.error("No successful generations to analyze")
            return
            
        # Calculate statistics
        processing_times = [r["processing_time"] for r in successful_results]
        audio_durations = [r["audio_duration"] for r in successful_results]
        rtfs = [r["rtf"] for r in successful_results]
        
        stats = {
            "total_tasks": len(self.results),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "exceptions": len(exceptions),
            "total_audio_duration": sum(audio_durations),
            "avg_processing_time": np.mean(processing_times),
            "median_processing_time": np.median(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "std_processing_time": np.std(processing_times),
            "avg_rtf": np.mean(rtfs),
            "median_rtf": np.median(rtfs),
            "min_rtf": min(rtfs),
            "max_rtf": max(rtfs)
        }
        
        # Print detailed statistics
        logger.info("\n" + "="*50)
        logger.info("BENCHMARK RESULTS")
        logger.info("="*50)
        logger.info(f"Total tasks: {stats['total_tasks']}")
        logger.info(f"Successful tasks: {stats['successful_tasks']} ({stats['successful_tasks']/stats['total_tasks']*100:.1f}%)")
        logger.info(f"Failed tasks: {stats['failed_tasks']}")
        logger.info(f"Exceptions: {stats['exceptions']}")
        logger.info(f"Total audio generated: {stats['total_audio_duration']:.2f} seconds")
        logger.info("\nProcessing Time Statistics (seconds):")
        logger.info(f"  Average: {stats['avg_processing_time']:.2f}")
        logger.info(f"  Median: {stats['median_processing_time']:.2f}")
        logger.info(f"  Min: {stats['min_processing_time']:.2f}")
        logger.info(f"  Max: {stats['max_processing_time']:.2f}")
        logger.info(f"  Std Dev: {stats['std_processing_time']:.2f}")
        logger.info("\nReal-Time Factor Statistics (lower is better):")
        logger.info(f"  Average RTF: {stats['avg_rtf']:.3f}")
        logger.info(f"  Median RTF: {stats['median_rtf']:.3f}")
        logger.info(f"  Best RTF: {stats['min_rtf']:.3f}")
        logger.info(f"  Worst RTF: {stats['max_rtf']:.3f}")
        logger.info("="*50)
        
        # Save results to file
        with open(os.path.join(self.output_dir, "benchmark_stats.txt"), "w") as f:
            f.write("ORPHEUS TTS BENCHMARK RESULTS\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Concurrent tasks: {self.concurrent_tasks}\n\n")
            
            f.write(f"Total tasks: {stats['total_tasks']}\n")
            f.write(f"Successful tasks: {stats['successful_tasks']} ({stats['successful_tasks']/stats['total_tasks']*100:.1f}%)\n")
            f.write(f"Failed tasks: {stats['failed_tasks']}\n")
            f.write(f"Exceptions: {stats['exceptions']}\n")
            f.write(f"Total audio generated: {stats['total_audio_duration']:.2f} seconds\n\n")
            
            f.write("Processing Time Statistics (seconds):\n")
            f.write(f"  Average: {stats['avg_processing_time']:.2f}\n")
            f.write(f"  Median: {stats['median_processing_time']:.2f}\n")
            f.write(f"  Min: {stats['min_processing_time']:.2f}\n")
            f.write(f"  Max: {stats['max_processing_time']:.2f}\n")
            f.write(f"  Std Dev: {stats['std_processing_time']:.2f}\n\n")
            
            f.write("Real-Time Factor Statistics (lower is better):\n")
            f.write(f"  Average RTF: {stats['avg_rtf']:.3f}\n")
            f.write(f"  Median RTF: {stats['median_rtf']:.3f}\n")
            f.write(f"  Best RTF: {stats['min_rtf']:.3f}\n")
            f.write(f"  Worst RTF: {stats['max_rtf']:.3f}\n")
        
        return stats

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark OrpheusTTS concurrent generation capabilities")
    parser.add_argument("--model", type=str, default="canopylabs/orpheus-tts-0.1-finetune-prod",
                        help="Model name or path")
    parser.add_argument("--concurrent", type=int, default=200,
                        help="Number of concurrent generation tasks")
    args = parser.parse_args()
    
    logger.info(f"Starting OrpheusTTS benchmark with {args.concurrent} concurrent tasks")
    
    benchmark = TTSBenchmark(model_name=args.model, concurrent_tasks=args.concurrent)
    benchmark.run_benchmark()
    benchmark.analyze_results()
    
if __name__ == "__main__":
    main()
