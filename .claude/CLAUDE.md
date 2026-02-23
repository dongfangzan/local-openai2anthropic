# 开发指南

## 配置文件

- 位置: `~/.oa2a/config.toml`（Linux/macOS）或 `%USERPROFILE%\.oa2a\config.toml`（Windows）
- 首次运行自动创建，编辑添加 API Key

## 发布流程

1. 更新版本号（pyproject.toml, `__init__.py`, main.py）
2. 运行测试: `pytest`
3. 提交并推送代码
4. 创建并推送 tag 触发 GitHub Actions 发布:
   ```bash
   git tag vx.y.z
   git push origin vx.y.z
   ```

## 提交规范

- 新代码测试覆盖率 > 90%
- 总体覆盖率 ≥ 80%
- 运行 `superpowers:verification-before-completion` 验证通过后再提交
