# Zsh 설정 가이드

이 문서는 Oh My Zsh와 agnoster 테마를 사용한 zsh 설정을 다른 PC에서 동일하게 적용하기 위한 가이드입니다.

## 목차
- [필수 요구사항](#필수-요구사항)
- [설치 단계](#설치-단계)
- [설정 파일](#설정-파일)
- [자동 설치 스크립트](#자동-설치-스크립트)
- [문제 해결](#문제-해결)

## 필수 요구사항

- Linux 또는 macOS 환경
- curl 또는 wget 설치
- git 설치
- 인터넷 연결

## 설치 단계

### 1. Zsh 설치

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install zsh
```

#### Linux (CentOS/RHEL/Fedora)
```bash
# CentOS/RHEL
sudo yum install zsh

# Fedora
sudo dnf install zsh
```

#### macOS
```bash
# Homebrew 사용
brew install zsh
```

#### sudo 권한이 없는 환경
```bash
# zsh-bin을 사용한 사용자 영역 설치
mkdir -p ~/.local/bin
wget -qO- https://raw.githubusercontent.com/romkatv/zsh-bin/master/install | bash -s -- -d ~/.local -e no
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Oh My Zsh 설치

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

### 3. 유용한 플러그인 설치

```bash
# zsh-autosuggestions (자동 완성 제안)
git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions

# zsh-syntax-highlighting (구문 강조)
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting
```

### 4. .zshrc 설정

`~/.zshrc` 파일을 다음과 같이 수정합니다:

```bash
# PATH 설정 (sudo 권한이 없는 환경에서 zsh-bin 사용 시)
export PATH=$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH

# Oh My Zsh 경로
export ZSH="$HOME/.oh-my-zsh"

# 테마 설정 (agnoster)
ZSH_THEME="agnoster"

# 플러그인 설정
plugins=(
    git
    colored-man-pages
    command-not-found
    zsh-autosuggestions
    zsh-syntax-highlighting
)

# Oh My Zsh 로드
source $ZSH/oh-my-zsh.sh
```

### 5. 기본 셸로 설정

#### 방법 1: chsh 명령어 사용 (권한 필요)
```bash
# zsh 경로 확인
which zsh

# 기본 셸 변경
chsh -s $(which zsh)
```

#### 방법 2: .bashrc 수정 (권한 불필요)
`~/.bashrc` 파일 끝에 다음 코드 추가:

```bash
# Auto-switch to zsh
if [ -f ~/.local/bin/zsh ] && [ "$0" != "-zsh" ] && [ -z "$ZSH_VERSION" ]; then
    exec ~/.local/bin/zsh
fi
```

## 설정 파일

### .zshrc 전체 설정
```bash
# PATH 설정
export PATH=$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH

# Oh My Zsh 설정
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="agnoster"

# 플러그인
plugins=(
    git
    colored-man-pages
    command-not-found
    zsh-autosuggestions
    zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

# 사용자 정의 별칭 (선택사항)
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias zshconfig="nano ~/.zshrc"
alias ohmyzsh="nano ~/.oh-my-zsh"

# 환경 변수
export EDITOR='nano'
export LANG=en_US.UTF-8
```

## 자동 설치 스크립트

다음 스크립트를 `setup-zsh.sh`로 저장하여 자동 설치할 수 있습니다:

```bash
#!/bin/bash

echo "🚀 Zsh 설정 시작..."

# zsh 설치 확인
if ! command -v zsh &> /dev/null; then
    echo "📦 zsh 설치 중..."
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y zsh
    elif command -v yum &> /dev/null; then
        sudo yum install -y zsh
    elif command -v brew &> /dev/null; then
        brew install zsh
    else
        echo "⚠️  수동으로 zsh를 설치하거나 zsh-bin을 사용하세요"
        wget -qO- https://raw.githubusercontent.com/romkatv/zsh-bin/master/install | bash -s -- -d ~/.local -e no
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# Oh My Zsh 설치
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    echo "🎨 Oh My Zsh 설치 중..."
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
fi

# 플러그인 설치
echo "🔌 플러그인 설치 중..."
git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions 2>/dev/null || echo "zsh-autosuggestions 이미 설치됨"
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting 2>/dev/null || echo "zsh-syntax-highlighting 이미 설치됨"

# .zshrc 백업 및 설정
echo "⚙️  .zshrc 설정 중..."
cp ~/.zshrc ~/.zshrc.backup

cat > ~/.zshrc << 'EOF'
# PATH 설정
export PATH=$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH

# Oh My Zsh 설정
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="agnoster"

# 플러그인
plugins=(
    git
    colored-man-pages
    command-not-found
    zsh-autosuggestions
    zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

# 사용자 정의 설정
export EDITOR='nano'
export LANG=en_US.UTF-8

# 별칭
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias zshconfig="nano ~/.zshrc"
EOF

# bashrc에 zsh 자동 실행 추가
if ! grep -q "exec.*zsh" ~/.bashrc 2>/dev/null; then
    echo "🔄 .bashrc에 zsh 자동 실행 추가..."
    cat >> ~/.bashrc << 'EOF'

# Auto-switch to zsh
if [ -f ~/.local/bin/zsh ] && [ "$0" != "-zsh" ] && [ -z "$ZSH_VERSION" ]; then
    exec ~/.local/bin/zsh
elif command -v zsh &> /dev/null && [ "$0" != "-zsh" ] && [ -z "$ZSH_VERSION" ]; then
    exec zsh
fi
EOF
fi

echo "✅ Zsh 설정 완료!"
echo "🔄 새 터미널을 열거나 'source ~/.zshrc'를 실행하세요"
```

스크립트 실행:
```bash
chmod +x setup-zsh.sh
./setup-zsh.sh
```

## 문제 해결

### 1. agnoster 테마가 제대로 표시되지 않는 경우
- Powerline 폰트 설치 필요:
```bash
# Ubuntu/Debian
sudo apt install fonts-powerline

# macOS
brew install --cask font-meslo-lg-nerd-font
```

### 2. 플러그인이 작동하지 않는 경우
- 플러그인이 올바른 위치에 설치되었는지 확인:
```bash
ls -la ~/.oh-my-zsh/custom/plugins/
```
- .zshrc의 plugins 배열에 올바르게 추가되었는지 확인

### 3. zsh가 기본 셸로 설정되지 않는 경우
```bash
# 현재 셸 확인
echo $SHELL

# 사용 가능한 셸 목록 확인
cat /etc/shells

# 수동으로 기본 셸 변경
chsh -s /usr/bin/zsh  # 또는 $(which zsh)
```

### 4. 권한 없이 설치한 zsh 사용 시
```bash
# PATH에 ~/.local/bin이 포함되어 있는지 확인
echo $PATH

# .zshrc와 .bashrc에 PATH 설정이 올바른지 확인
grep PATH ~/.zshrc ~/.bashrc
```

### 5. 설정 초기화
```bash
# Oh My Zsh 재설치
rm -rf ~/.oh-my-zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# 백업된 설정 복원
cp ~/.zshrc.backup ~/.zshrc
```

## 추가 커스터마이징

### 다른 테마 사용
```bash
# 사용 가능한 테마 목록 확인
ls ~/.oh-my-zsh/themes/

# .zshrc에서 ZSH_THEME 변경
ZSH_THEME="powerlevel10k/powerlevel10k"  # 예시
```

### 추가 유용한 플러그인
- `z`: 자주 사용하는 디렉토리로 빠른 이동
- `extract`: 다양한 압축 파일 자동 해제
- `web-search`: 터미널에서 웹 검색
- `history-substring-search`: 히스토리 부분 문자열 검색

플러그인 추가 방법:
```bash
plugins=(
    git
    colored-man-pages
    command-not-found
    zsh-autosuggestions
    zsh-syntax-highlighting
    z
    extract
    web-search
    history-substring-search
)
```

---

이 가이드를 따라하면 동일한 zsh 환경을 다른 PC에서도 구축할 수 있습니다.