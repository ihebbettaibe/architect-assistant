#!/bin/bash

# Find all files in git history bigger than 50MB
echo "Detecting large files (>50MB) in git history..."

# List files larger than 50MB (use git rev-list + git cat-file)
large_files=$(git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '$1=="blob" && $3 > 50000000 {print $4}')

if [ -z "$large_files" ]; then
  echo "No large files detected."
  exit 0
fi

echo "Large files to remove:"
echo "$large_files"

# Build git filter-repo command
cmd="git filter-repo"

for file in $large_files; do
  cmd+=" --path $file --invert-paths"
done

echo "Running:"
echo $cmd

# Run the filter-repo command
$cmd

# Force push to origin main branch
echo "Force pushing cleaned repo..."
git push origin main --force

echo "Done cleaning and pushing."
