"""
プロンプト生成器

このモジュールは、実験用の全プロンプトパターンを自動生成します。
4×4×4=64パターンの組み合わせを管理します。

Classes:
    PromptGenerator: プロンプト生成とメタデータ管理

Example:
    >>> from src.prompt_generator import PromptGenerator
    >>> generator = PromptGenerator()
    >>> prompts = generator.generate_all_prompts()
    >>> print(f"生成されたプロンプト数: {len(prompts)}")
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
from itertools import product

import yaml

# ロガーの設定
logger = logging.getLogger(__name__)


class PromptGenerator:
    """
    プロンプト生成器

    設定ファイルから実験パラメータを読み込み、
    すべての組み合わせのプロンプトを生成します。

    Attributes:
        config (Dict[str, Any]): 実験設定
        models (List[Dict]): LLMモデルリスト
        personas (List[Dict]): ペルソナリスト
        travel_types (List[Dict]): 旅行タイプリスト
        template (str): プロンプトテンプレート
    """

    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        """
        PromptGeneratorの初期化

        Args:
            config_path: 設定ファイルのパス

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
            ValueError: 設定ファイルの形式が不正な場合
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"設定ファイルが見つかりません: {config_path}"
            )

        # 設定ファイルの読み込み
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 実験パラメータの抽出
        self.models = self.config.get("models", [])
        self.personas = self.config.get("personas", [])
        self.travel_types = self.config.get("travel_types", [])
        self.template = self.config.get("prompt_template", "")

        # バリデーション
        self._validate_config()

        logger.info(
            f"プロンプト生成器を初期化しました: "
            f"モデル={len(self.models)}, "
            f"ペルソナ={len(self.personas)}, "
            f"旅行タイプ={len(self.travel_types)}"
        )

    def _validate_config(self):
        """
        設定ファイルのバリデーション

        Raises:
            ValueError: 必須項目が不足している場合
        """
        if not self.models:
            raise ValueError("モデルが設定されていません")

        if not self.personas:
            raise ValueError("ペルソナが設定されていません")

        if not self.travel_types:
            raise ValueError("旅行タイプが設定されていません")

        if not self.template:
            raise ValueError("プロンプトテンプレートが設定されていません")

        # テンプレートのプレースホルダーチェック
        if "{persona}" not in self.template or "{travel_type}" not in self.template:
            raise ValueError(
                "プロンプトテンプレートに必要なプレースホルダー"
                "（{persona}, {travel_type}）が含まれていません"
            )

    def generate_prompt(
        self,
        persona_name: str,
        travel_type_name: str
    ) -> str:
        """
        単一のプロンプトを生成

        Args:
            persona_name: ペルソナ名
            travel_type_name: 旅行タイプ名

        Returns:
            生成されたプロンプト文字列
        """
        prompt = self.template.format(
            persona=persona_name,
            travel_type=travel_type_name
        )
        return prompt.strip()

    def generate_all_prompts(self) -> List[Dict[str, Any]]:
        """
        全パターンのプロンプトを生成

        4×4×4=64パターンのすべての組み合わせを生成します。

        Returns:
            プロンプト情報の辞書リスト。各辞書には以下のキーが含まれます:
                - experiment_id: 実験識別子（例: "exp_000"）
                - model: モデル名
                - model_display_name: モデル表示名
                - persona_id: ペルソナID
                - persona_name: ペルソナ名
                - travel_type_id: 旅行タイプID
                - travel_type_name: 旅行タイプ名
                - prompt: 生成されたプロンプト
                - metadata: メタデータ
        """
        prompts = []
        experiment_counter = 0

        # 3重ループですべての組み合わせを生成
        for model, persona, travel_type in product(
            self.models,
            self.personas,
            self.travel_types
        ):
            # プロンプト生成
            prompt_text = self.generate_prompt(
                persona_name=persona["name"],
                travel_type_name=travel_type["name"]
            )

            # 実験情報の構築
            prompt_info = {
                "experiment_id": f"exp_{experiment_counter:03d}",
                "model": model["name"],
                "model_display_name": model.get("display_name", model["name"]),
                "persona_id": persona["id"],
                "persona_name": persona["name"],
                "travel_type_id": travel_type["id"],
                "travel_type_name": travel_type["name"],
                "prompt": prompt_text,
                "metadata": {
                    "model_description": model.get("description", ""),
                    "persona_description": persona.get("description", ""),
                    "persona_characteristics": persona.get("characteristics", []),
                    "travel_type_description": travel_type.get("description", ""),
                    "target_audience": travel_type.get("target_audience", [])
                }
            }

            prompts.append(prompt_info)
            experiment_counter += 1

        logger.info(f"{len(prompts)}個のプロンプトを生成しました")
        return prompts

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        実験の要約情報を取得

        Returns:
            実験要約の辞書:
                - total_experiments: 総実験数
                - models_count: モデル数
                - personas_count: ペルソナ数
                - travel_types_count: 旅行タイプ数
                - models: モデルリスト
                - personas: ペルソナリスト
                - travel_types: 旅行タイプリスト
        """
        return {
            "total_experiments": len(self.models) * len(self.personas) * len(self.travel_types),
            "models_count": len(self.models),
            "personas_count": len(self.personas),
            "travel_types_count": len(self.travel_types),
            "models": [m["name"] for m in self.models],
            "personas": [p["name"] for p in self.personas],
            "travel_types": [t["name"] for t in self.travel_types],
        }

    def filter_prompts(
        self,
        prompts: List[Dict[str, Any]],
        model: str = None,
        persona_id: str = None,
        travel_type_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        プロンプトをフィルタリング

        Args:
            prompts: プロンプトリスト
            model: フィルタするモデル名（オプション）
            persona_id: フィルタするペルソナID（オプション）
            travel_type_id: フィルタする旅行タイプID（オプション）

        Returns:
            フィルタされたプロンプトリスト
        """
        filtered = prompts

        if model:
            filtered = [p for p in filtered if p["model"] == model]

        if persona_id:
            filtered = [p for p in filtered if p["persona_id"] == persona_id]

        if travel_type_id:
            filtered = [p for p in filtered if p["travel_type_id"] == travel_type_id]

        logger.info(
            f"フィルタリング結果: {len(filtered)}個のプロンプト "
            f"(元: {len(prompts)}個)"
        )

        return filtered

    def export_prompts_to_text(
        self,
        prompts: List[Dict[str, Any]],
        output_path: str = "data/prompts_list.txt"
    ):
        """
        プロンプトリストをテキストファイルにエクスポート

        Args:
            prompts: プロンプトリスト
            output_path: 出力ファイルパス
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("LLM地域観光紹介性能比較実験 - プロンプト一覧\n")
            f.write("=" * 80 + "\n\n")

            for prompt_info in prompts:
                f.write(f"実験ID: {prompt_info['experiment_id']}\n")
                f.write(f"モデル: {prompt_info['model_display_name']}\n")
                f.write(f"ペルソナ: {prompt_info['persona_name']}\n")
                f.write(f"旅行タイプ: {prompt_info['travel_type_name']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{prompt_info['prompt']}\n")
                f.write("=" * 80 + "\n\n")

        logger.info(f"プロンプトリストを出力しました: {output_path}")


if __name__ == "__main__":
    # モジュール単体実行時のテスト
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # プロンプト生成器の初期化
        generator = PromptGenerator()

        # 実験要約の表示
        summary = generator.get_experiment_summary()
        print("\n" + "=" * 60)
        print("実験要約")
        print("=" * 60)
        print(f"総実験数: {summary['total_experiments']}")
        print(f"モデル数: {summary['models_count']}")
        print(f"ペルソナ数: {summary['personas_count']}")
        print(f"旅行タイプ数: {summary['travel_types_count']}")

        # 全プロンプト生成
        all_prompts = generator.generate_all_prompts()
        print(f"\n✓ {len(all_prompts)}個のプロンプトを生成しました")

        # サンプル表示
        print("\n" + "=" * 60)
        print("サンプルプロンプト（最初の2つ）")
        print("=" * 60)
        for i, prompt in enumerate(all_prompts[:2]):
            print(f"\n[{i+1}] {prompt['experiment_id']}")
            print(f"モデル: {prompt['model_display_name']}")
            print(f"ペルソナ: {prompt['persona_name']}")
            print(f"旅行タイプ: {prompt['travel_type_name']}")
            print(f"\nプロンプト:\n{prompt['prompt']}")
            print("-" * 60)

        # テキストファイルへのエクスポート
        generator.export_prompts_to_text(all_prompts)

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
