
# Git Workflow for Forking and Syncing with Upstream (SSH)

This guide explains how to fork the original repository, switch to SSH, and keep your fork in sync.

![Git Workflow Diagram](2bc8ae0891.png)

## 1. Fork the Repository on GitHub
Original repo: https://github.com/kspaceKelvin/python-ismrmrd-server

Fork it to your GitHub account: https://github.com/RalfLoeffler

## 2. Update Local Clone to Use SSH and Your Fork
```
git remote set-url origin git@github.com:RalfLoeffler/python-ismrmrd-server.git
```

## 3. Add Original Repository as `upstream`
```
git remote add upstream git@github.com:kspaceKelvin/python-ismrmrd-server.git
```

## 4. Verify Remotes
```
git remote -v
```

## 5. Sync with Upstream
```
git fetch upstream
git checkout master
git merge upstream/master
# or: git rebase upstream/master

git push origin master
```

