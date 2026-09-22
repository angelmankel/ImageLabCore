"""
CLI commands for the external-api workflow schema system.
"""
import argparse
from pathlib import Path
from typing import List, Optional
import yaml

# Support both relative and absolute imports
try:
    from .schema_generator import (
        generate_workflow_schema,
        validate_workflow_schema,
    )
except ImportError:
    from schema_generator import (
        generate_workflow_schema,
        validate_workflow_schema,
    )


def get_workflows_dir() -> Path:
    """Get the workflows directory."""
    return Path(__file__).parent / "workflows"


def find_workflow_files(workflows_dir: Path) -> List[Path]:
    """Find all workflow .py files (excluding __init__.py and base files)."""
    workflow_files = []
    for py_file in workflows_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        if py_file.name == "base.py":
            continue
        workflow_files.append(py_file)
    return sorted(workflow_files)


def format_schema_yaml(schema: dict) -> str:
    """Format schema as nicely-formatted YAML string (matches write_yaml output)."""
    lines = []
    
    # Metadata section
    lines.append(f"name: {schema['name']}")
    lines.append(f"description: {schema.get('description', '')}")
    lines.append(f"category: {schema.get('category', 'Uncategorized')}")
    
    # Model types
    model_types = schema.get('model_types', [])
    if model_types:
        lines.append("model_types:")
        for mt in model_types:
            lines.append(f"  - {mt}")
    else:
        lines.append("model_types: []")
    
    # Double newline before parameters
    lines.append("")
    lines.append("parameters:")
    
    # Format each parameter with a blank line between them
    parameters = schema.get('parameters', [])
    for i, param in enumerate(parameters):
        # Add blank line between parameters (not before first one)
        if i > 0:
            lines.append("")
        
        # Dump the parameter to YAML and indent it
        param_yaml = yaml.dump([param], default_flow_style=False, allow_unicode=True, sort_keys=False)
        # Remove the leading "- " and adjust indentation
        param_lines = param_yaml.strip().split('\n')
        for line in param_lines:
            lines.append(f"  {line}")
    
    return '\n'.join(lines)


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate YAML schema(s) from workflow files."""
    workflows_dir = get_workflows_dir()
    
    if args.all:
        # Generate for all workflows
        workflow_files = find_workflow_files(workflows_dir)
        if not workflow_files:
            print("No workflow files found.")
            return 1
        
        for wf_path in workflow_files:
            try:
                schema = generate_workflow_schema(wf_path, dry_run=args.dry_run)
                action = "Would generate" if args.dry_run else "Generated"
                yaml_path = wf_path.with_suffix(".yaml")
                print(f"{action}: {yaml_path.name}")
                
                if args.dry_run:
                    print(format_schema_yaml(schema))
                    print("-" * 40)
            except Exception as e:
                print(f"Error processing {wf_path.name}: {e}")
        
        return 0
    
    elif args.workflow:
        # Generate for single workflow
        workflow_path = workflows_dir / f"{args.workflow}.py"
        if not workflow_path.exists():
            print(f"Workflow not found: {workflow_path}")
            return 1
        
        try:
            schema = generate_workflow_schema(workflow_path, dry_run=args.dry_run)
            yaml_path = workflow_path.with_suffix(".yaml")
            
            if args.dry_run:
                print(f"Would generate: {yaml_path.name}")
                print(format_schema_yaml(schema))
            else:
                print(f"Generated: {yaml_path.name}")
            
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    else:
        print("Must specify --all or a workflow name")
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate YAML schema(s) against workflow files."""
    workflows_dir = get_workflows_dir()
    all_valid = True
    
    if args.all:
        workflow_files = find_workflow_files(workflows_dir)
    elif args.workflow:
        workflow_path = workflows_dir / f"{args.workflow}.py"
        if not workflow_path.exists():
            print(f"Workflow not found: {workflow_path}")
            return 1
        workflow_files = [workflow_path]
    else:
        print("Must specify --all or a workflow name")
        return 1
    
    for wf_path in workflow_files:
        errors = validate_workflow_schema(wf_path)
        
        if errors:
            all_valid = False
            print(f"❌ {wf_path.name}:")
            for error in errors:
                print(f"   - {error}")
        else:
            print(f"✓ {wf_path.name}")
    
    return 0 if all_valid else 1


def cmd_list(args: argparse.Namespace) -> int:
    """List all workflows and their YAML status."""
    workflows_dir = get_workflows_dir()
    workflow_files = find_workflow_files(workflows_dir)
    
    if not workflow_files:
        print("No workflow files found.")
        return 1
    
    print(f"Workflows in {workflows_dir}:\n")
    
    for wf_path in workflow_files:
        yaml_path = wf_path.with_suffix(".yaml")
        has_yaml = yaml_path.exists()
        
        status = "✓ YAML" if has_yaml else "✗ no YAML"
        print(f"  {wf_path.stem:30} [{status}]")
    
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Watch for changes and auto-regenerate YAML schemas."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    except ImportError:
        print("Error: watchdog not installed.")
        print("Install it with: pip install watchdog")
        return 1
    
    workflows_dir = get_workflows_dir()
    
    class WorkflowChangeHandler(FileSystemEventHandler):
        def __init__(self):
            self.last_modified = {}
        
        def on_modified(self, event):
            self._handle_change(event)
        
        def on_created(self, event):
            self._handle_change(event)
        
        def _handle_change(self, event):
            if event.is_directory:
                return
            
            src_path = Path(event.src_path)
            
            # Only process .py files
            if src_path.suffix != ".py":
                return
            
            # Skip special files
            if src_path.name.startswith("_") or src_path.name == "base.py":
                return
            
            # Debounce: skip if modified within last second
            import time
            now = time.time()
            last = self.last_modified.get(str(src_path), 0)
            if now - last < 1.0:
                return
            self.last_modified[str(src_path)] = now
            
            # Regenerate YAML
            print(f"\n[watch] Detected change: {src_path.name}")
            try:
                schema = generate_workflow_schema(src_path)
                yaml_path = src_path.with_suffix(".yaml")
                print(f"[watch] Regenerated: {yaml_path.name}")
            except Exception as e:
                print(f"[watch] Error: {e}")
    
    print(f"Watching for changes in: {workflows_dir}")
    print("Press Ctrl+C to stop.\n")
    
    # List current workflows
    workflow_files = find_workflow_files(workflows_dir)
    for wf_path in workflow_files:
        yaml_path = wf_path.with_suffix(".yaml")
        status = "✓" if yaml_path.exists() else "✗"
        print(f"  {status} {wf_path.stem}")
    print()
    
    event_handler = WorkflowChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, str(workflows_dir), recursive=False)
    observer.start()
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[watch] Stopped.")
    
    observer.join()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="external-api",
        description="External API workflow schema management"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate YAML schema from workflow file"
    )
    gen_parser.add_argument(
        "workflow",
        nargs="?",
        help="Workflow name (without .py extension)"
    )
    gen_parser.add_argument(
        "--all",
        action="store_true",
        help="Generate schemas for all workflows"
    )
    gen_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview generated schema without writing"
    )
    gen_parser.set_defaults(func=cmd_generate)
    
    # validate command
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate YAML schema matches workflow"
    )
    val_parser.add_argument(
        "workflow",
        nargs="?",
        help="Workflow name (without .py extension)"
    )
    val_parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all workflows"
    )
    val_parser.set_defaults(func=cmd_validate)
    
    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List all workflows"
    )
    list_parser.set_defaults(func=cmd_list)
    
    # watch command
    watch_parser = subparsers.add_parser(
        "watch",
        help="Watch for changes and auto-regenerate YAML (requires: pip install watchdog)"
    )
    watch_parser.set_defaults(func=cmd_watch)
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
