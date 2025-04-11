# Git Branch Management

## Deleting Branches

### Delete a Local Branch

```bash
# Delete a local branch (if fully merged to current branch)
git branch -d branch-name

# Force delete a local branch (even if not merged)
git branch -D branch-name
```

### Delete a Remote Branch

```bash
# Delete a remote branch
git push origin --delete branch-name
```

### Complete Workflow for Branch Deletion

```bash
# Step 1: Switch to a different branch (typically main/master)
git checkout main

# Step 2: Delete the local branch
git branch -d branch-name
# Or force delete if needed
# git branch -D branch-name

# Step 3: Delete the remote branch
git push origin --delete branch-name

# Step 4: Verify branches are gone
# Check local branches
git branch

# Check remote branches
git branch -r
```

## Additional Branch Management Commands

### List Branches

```bash
# List local branches
git branch

# List remote branches
git branch -r

# List all branches (local and remote)
git branch -a
```

### Prune Deleted Remote Branches

If others have deleted remote branches, update your local references:

```bash
# Prune references to deleted remote branches
git fetch --prune
```
