import os
import logging
import argparse
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
from rich.progress import Progress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rich console for better output formatting
console = Console()

class TextCompleter:
    """Class to handle text completion using transformer models."""
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """
        Initialize the text completer with specified model.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model and tokenizer
        with console.status(f"[bold green]Loading {model_name} model...[/]"):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.device)
                logger.info(f"Model {model_name} loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    def complete(self, 
                prompt: str, 
                max_length: int = 50, 
                num_return_sequences: int = 1,
                temperature: float = 1.0,
                top_p: float = 0.9,
                top_k: int = 50,
                repetition_penalty: float = 1.0,
                seed: Optional[int] = 42) -> List[str]:
        """
        Generate completions for the given prompt.
        
        Args:
            prompt: The text to complete
            max_length: Maximum length of the output (including prompt)
            num_return_sequences: Number of different completions to generate
            temperature: Controls randomness (lower = more deterministic)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            seed: Random seed for reproducibility
            
        Returns:
            List of generated completions
        """
        if seed is not None:
            set_seed(seed)
            
        try:
            # Encode the input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate completions
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode outputs
            completions = []
            for sequence in output_sequences:
                completion = self.tokenizer.decode(sequence, skip_special_tokens=True)
                completions.append(completion)
                
            return completions
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise

def interactive_mode(completer: TextCompleter, max_length: int = 50, 
                    temperature: float = 0.8, num_sequences: int = 1):
    """Run the text completer in interactive mode."""
    console.print(Panel.fit(
        "[bold blue]Text Completion Generator[/bold blue]\n"
        "Type 'exit' or 'quit' to end the session.\n"
        "Type 'settings' to view or change generation settings.",
        title="Interactive Mode",
        border_style="green"
    ))
    
    settings = {
        "max_length": max_length,
        "temperature": temperature,
        "num_sequences": num_sequences,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0
    }
    
    while True:
        try:
            # Get user input
            prompt = console.input("\n[bold yellow]Enter text to complete:[/] ")
            
            # Check for exit command
            if prompt.lower() in ["exit", "quit"]:
                console.print("[bold]Goodbye![/]")
                break
                
            # Check for settings command
            elif prompt.lower() == "settings":
                _show_and_update_settings(settings)
                continue
                
            # Handle empty input
            elif not prompt.strip():
                console.print("[bold red]Please enter some text.[/]")
                continue
                
            # Generate completions
            with console.status("[bold green]Generating completions...[/]"):
                completions = completer.complete(
                    prompt=prompt,
                    max_length=settings["max_length"],
                    num_return_sequences=settings["num_sequences"],
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    top_k=settings["top_k"],
                    repetition_penalty=settings["repetition_penalty"]
                )
            
            # Display results
            console.print("\n[bold green]Completions:[/]")
            for i, completion in enumerate(completions, 1):
                console.print(Panel(
                    completion,
                    title=f"Completion {i}/{settings['num_sequences']}",
                    border_style="blue"
                ))
                
        except KeyboardInterrupt:
            console.print("\n[bold]Session interrupted. Goodbye![/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")

def _show_and_update_settings(settings: Dict[str, Any]) -> None:
    """Show current settings and allow user to update them."""
    console.print("\n[bold blue]Current Settings:[/]")
    for key, value in settings.items():
        console.print(f"  [yellow]{key}:[/] {value}")
    
    console.print("\n[bold]Enter setting to change (or empty to return):[/]")
    setting = console.input("> ").lower().strip()
    
    if not setting:
        return
        
    if setting in settings:
        try:
            new_value = console.input(f"Enter new value for [yellow]{setting}[/] (current: {settings[setting]}): ")
            
            # Convert value to appropriate type
            if isinstance(settings[setting], int):
                settings[setting] = int(new_value)
            elif isinstance(settings[setting], float):
                settings[setting] = float(new_value)
            else:
                settings[setting] = new_value
                
            console.print(f"[green]Updated {setting} to {settings[setting]}[/]")
        except ValueError:
            console.print("[bold red]Invalid value. Setting unchanged.[/]")
    else:
        console.print("[bold red]Setting not found.[/]")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Text Completion Generator")
    
    # Model selection
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt2",
        help="Name of the model to use (default: gpt2)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=50,
        help="Maximum length of generated text (default: 50)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8,
        help="Temperature for generation (default: 0.8)"
    )
    parser.add_argument(
        "--num-sequences", 
        type=int, 
        default=1,
        help="Number of completions to generate (default: 1)"
    )
    
    # Run modes
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--prompt", 
        type=str,
        help="Text prompt to complete (non-interactive mode)"
    )
    
    # Device selection
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Initialize the completer
        completer = TextCompleter(model_name=args.model, device=args.device)
        
        # Run in appropriate mode
        if args.interactive:
            interactive_mode(
                completer, 
                max_length=args.max_length,
                temperature=args.temperature,
                num_sequences=args.num_sequences
            )
        elif args.prompt:
            # Single completion mode
            with console.status("[bold green]Generating completions...[/]"):
                completions = completer.complete(
                    prompt=args.prompt,
                    max_length=args.max_length,
                    num_return_sequences=args.num_sequences,
                    temperature=args.temperature
                )
            
            # Display results
            console.print("\n[bold green]Completions for:[/]", args.prompt)
            for i, completion in enumerate(completions, 1):
                console.print(Panel(
                    completion,
                    title=f"Completion {i}/{args.num_sequences}",
                    border_style="blue"
                ))
        else:
            # No mode specified
            console.print("[bold red]Error:[/] Please specify either --interactive or --prompt")
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")

if __name__ == "__main__":
    main()